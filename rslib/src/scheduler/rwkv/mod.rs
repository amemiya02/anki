// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Hayaki: the opt-in RWKV review queue (Hayaki ADR-0009).
//!
//! With the collection config key [CONFIG_KEY] set to true and the model
//! ready, review cards in their home deck are queued when the model's
//! immediate head predicts that their chance of recall has fallen to the
//! deck's desired retention. Everything else, including answering, stays
//! upstream's. While the model is not ready, the queue is upstream's.
//!
//! The model state is derived from the revlog and never synced. Each device
//! replays the revlog once, keeps a snapshot beside the collection, and then
//! follows new answers, undos and syncs through [Engine::catch_up].

pub(crate) mod history;

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use fushi_rwkv::Model;
use fushi_rwkv::ReviewInput;
use fushi_rwkv::RwkvRuntime;
use fushi_rwkv::StatePrecision;
use history::CardHistory;
use history::KeptRow;
use history::Mappings;
use history::RevlogRow;
use serde::Deserialize;
use serde::Serialize;

use super::queue::DueCard;
use super::queue::DueCardKind;
use super::timing::SchedTimingToday;
use crate::card::CardQueue;
use crate::prelude::*;

/// The collection config key that turns RWKV scheduling on. Absent means off.
pub const CONFIG_KEY: &str = "hayakiRwkvScheduling";

/// The small model `sm56` (JSchoreels/anki-rwkv), int8 weights. The weights are
/// for personal and research use; see WEIGHTS-NOTICE.md in that repository.
pub const WEIGHTS_FILE_NAME: &str = "rwkv-sm56-weights.int8.bin";
pub const WEIGHTS_URL: &str = "https://raw.githubusercontent.com/JSchoreels/anki/001440b03395f24ffc1714b18ad3d9d4d89d2067/qt/aqt/rwkv_inference/sm56/weights.int8.bin";
pub const WEIGHTS_SHA256: &str = "479b3e2bc6a1c900ba8a4a45d629965c6df2a20cc8ff294583f5aeaeea3b4fdd";
pub const WEIGHTS_LEN: usize = 436_391;

/// A card answered today comes back only after this long...
const MIN_ELAPSED_SECS: i64 = 30;
/// ...and after this many other answers (kuma3 defaults).
const MIN_INTERVENING_REVIEWS: u64 = 5;
/// Cached end-of-day scores are recomputed after this many answers, because
/// every answer moves the deck, preset and collection streams.
const REQUERY_EVERY_ANSWERS: u32 = 20;
const REPLAY_CHUNK: usize = 8_192;
const SNAPSHOT_MAGIC: &[u8; 8] = b"HYKRWKV2";
/// Undo history kept for catch-up, like the runtime's own journal (about
/// 600 entries).
const JOURNAL_CAP: usize = 512;
/// A failed engine prepares again after this long.
const RETRY_FAILED_AFTER: std::time::Duration = std::time::Duration::from_secs(600);

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Status {
    #[default]
    NotPrepared,
    Preparing {
        done: u64,
        total: u64,
    },
    Ready,
    Failed(String),
}

/// What the runtime has replayed, so changes to the collection can be
/// applied incrementally.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Replayed {
    /// Kept rows replayed so far, across the whole timeline.
    rows: u64,
    cards: HashMap<i64, CardHistory>,
    mappings: Mappings,
    signature: Signature,
    /// The time of day the scheduling day starts, in seconds. Day indexes
    /// depend on it, so a change replays in full.
    rollover_secs: i64,
    /// Revlog rows seen since the last full replay or restore, oldest first,
    /// so undoing them can be followed. Capped like the runtime's own journal.
    #[serde(skip)]
    journal: Vec<JournalEntry>,
}

#[derive(Debug, Clone)]
struct JournalEntry {
    review_id: i64,
    /// Set for rows appended to the runtime. Rows the model does not replay
    /// (rescheduling, set due date) are kept too, so undoing them is followed.
    appended: Option<Appended>,
}

#[derive(Debug, Clone, Copy)]
struct Appended {
    card_id: i64,
    previous: Option<CardHistory>,
}

#[derive(Debug, Clone, Copy)]
struct Score {
    day: i64,
    last_review_id: i64,
    note_id: i64,
    r: f32,
}

#[derive(Default)]
pub struct Engine {
    status: Status,
    failed_at: Option<Instant>,
    model: Option<Model>,
    runtime: Option<RwkvRuntime>,
    replayed: Replayed,
    scores: HashMap<i64, Score>,
    answers_since_requery: u32,
    /// Bumped whenever the engine becomes ready or replays in full, so cached
    /// study queues built under another state are rebuilt.
    pub(crate) epoch: u64,
    /// Bumped when a replay or restore starts and when the collection closes,
    /// so a thread working for an older state discards its result.
    replay_generation: u64,
    snapshot_path: Option<PathBuf>,
}

impl fmt::Debug for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Engine")
            .field("status", &self.status)
            .field("rows", &self.replayed.rows)
            .field("epoch", &self.epoch)
            .finish()
    }
}

pub type SharedEngine = Arc<Mutex<Engine>>;

/// The model the app loaded with PrepareRwkv in this process. A collection
/// opened later (another profile, or after a full sync) prepares itself from
/// it without another call.
static LOADED_MODEL: Mutex<Option<Model>> = Mutex::new(None);

pub fn set_loaded_model(model: Model) {
    *LOADED_MODEL.lock().unwrap() = Some(model);
}

fn loaded_model() -> Option<Model> {
    LOADED_MODEL.lock().unwrap().clone()
}

/// What a caller needs to know about one review card the RWKV rule queues.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RwkvDueCard {
    pub card: DueCard,
    pub retrievability: f32,
}

struct ReplayInput {
    revlog: Vec<RevlogRow>,
    mappings: Mappings,
    timing: SchedTimingToday,
    signature: Signature,
}

/// What catch-up compares to decide whether anything changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Signature {
    revlog_count: u64,
    revlog_max_id: i64,
    cards_max_mod: i64,
    cards_count: u64,
    /// Local deletions write graves until the next sync.
    graves_count: u64,
    /// A sync can bring cards with modification times older than local ones,
    /// so a changed sync time compares every card.
    last_sync: i64,
}

impl Signature {
    fn read(col: &Collection) -> Result<Self> {
        let db = &col.storage.db;
        let (revlog_count, revlog_max_id) = db.query_row(
            "select count(), coalesce(max(id), 0) from revlog",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let (cards_count, cards_max_mod) = db.query_row(
            "select count(), coalesce(max(mod), 0) from cards",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let graves_count = db.query_row("select count() from graves", [], |row| row.get(0))?;
        Ok(Self {
            revlog_count,
            revlog_max_id,
            cards_max_mod,
            cards_count,
            graves_count,
            last_sync: col.storage.get_collection_timestamps()?.last_sync.0,
        })
    }
}

impl Engine {
    pub fn status(&self) -> &Status {
        &self.status
    }

    fn is_ready(&self) -> bool {
        self.status == Status::Ready && self.runtime.is_some()
    }

    fn fail(&mut self, message: String) {
        self.status = Status::Failed(message);
        self.failed_at = Some(Instant::now());
        self.runtime = None;
        self.epoch += 1;
    }

    fn push_journal(&mut self, entry: JournalEntry) {
        self.replayed.journal.push(entry);
        if self.replayed.journal.len() > JOURNAL_CAP {
            self.replayed.journal.remove(0);
        }
    }

    /// Appends one kept row through the runtime's journal. Returns false if
    /// the runtime refused it, which the caller answers with a full replay.
    fn append(&mut self, row: &KeptRow, timing: &SchedTimingToday) -> bool {
        let Some(runtime) = self.runtime.as_mut() else {
            return false;
        };
        let previous = self.replayed.cards.get(&row.card_id).copied();
        let (input, history) =
            history::answered_input(row, previous.as_ref(), self.replayed.rows, timing);
        if runtime
            .append(&[input], &[row.review_id as u64], &[row.review_id])
            .is_err()
        {
            return false;
        }
        self.replayed.rows += 1;
        self.replayed.cards.insert(row.card_id, history);
        self.push_journal(JournalEntry {
            review_id: row.review_id,
            appended: Some(Appended {
                card_id: row.card_id,
                previous,
            }),
        });
        self.forget_note_scores(row.note_id);
        self.answers_since_requery += 1;
        true
    }

    /// Undoes the newest journal entry if it is `review_id`.
    fn undo_newest(&mut self, review_id: i64) -> bool {
        let Some(entry) = self.replayed.journal.last().cloned() else {
            return false;
        };
        if entry.review_id != review_id {
            return false;
        }
        if let Some(appended) = entry.appended {
            let Some(runtime) = self.runtime.as_mut() else {
                return false;
            };
            if !matches!(runtime.unappend(review_id as u64), Ok(true)) {
                return false;
            }
            self.replayed.rows -= 1;
            let note_id = self
                .replayed
                .cards
                .get(&appended.card_id)
                .map(|h| h.note_id)
                .unwrap_or_default();
            match appended.previous {
                Some(previous) => self.replayed.cards.insert(appended.card_id, previous),
                None => self.replayed.cards.remove(&appended.card_id),
            };
            self.forget_note_scores(note_id);
        }
        self.replayed.journal.pop();
        true
    }

    fn forget_note_scores(&mut self, note_id: i64) {
        self.scores.retain(|_, score| score.note_id != note_id);
    }

    /// The snapshot file's contents, taken under the lock. Writing happens
    /// outside it.
    fn snapshot_bytes(&self) -> Option<(PathBuf, Vec<u8>)> {
        let path = self.snapshot_path.clone()?;
        let state = self.runtime.as_ref()?.snapshot().ok()?;
        let replayed = serde_json::to_vec(&self.replayed).ok()?;
        let mut bytes = Vec::with_capacity(80 + replayed.len() + state.len());
        bytes.extend_from_slice(SNAPSHOT_MAGIC);
        bytes.extend_from_slice(WEIGHTS_SHA256.as_bytes());
        bytes.extend_from_slice(&(replayed.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&replayed);
        bytes.extend_from_slice(&state);
        Some((path, bytes))
    }

    fn read_snapshot(path: &Path, model: &Model) -> Option<(Replayed, RwkvRuntime)> {
        let bytes = std::fs::read(path).ok()?;
        let header = SNAPSHOT_MAGIC.len() + WEIGHTS_SHA256.len();
        if bytes.len() < header + 8
            || &bytes[..SNAPSHOT_MAGIC.len()] != SNAPSHOT_MAGIC
            || &bytes[SNAPSHOT_MAGIC.len()..header] != WEIGHTS_SHA256.as_bytes()
        {
            return None;
        }
        let len = u64::from_le_bytes(bytes[header..header + 8].try_into().ok()?) as usize;
        let body = header + 8;
        let replayed: Replayed = serde_json::from_slice(bytes.get(body..body + len)?).ok()?;
        let mut runtime = RwkvRuntime::new(model, StatePrecision::Int4, 0).ok()?;
        runtime.restore(bytes.get(body + len..)?).ok()?;
        Some((replayed, runtime))
    }

    /// Called when the collection closes: keeps the current state for the
    /// next open, and stops any replay thread from installing or writing.
    pub(crate) fn on_close(&mut self) {
        self.replay_generation += 1;
        if self.is_ready() {
            if let Some((path, bytes)) = self.snapshot_bytes() {
                write_snapshot_file(&path, &bytes);
            }
        }
    }
}

fn write_snapshot_file(path: &Path, bytes: &[u8]) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let tmp = path.with_extension(format!("rwkv.{}.{nanos}.tmp", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Replays `input` into a fresh runtime off the collection thread, then
/// installs it if no newer replay has started and the collection is open.
fn spawn_full_replay(shared: SharedEngine, model: Model, input: ReplayInput, generation: u64) {
    std::thread::spawn(move || {
        let result = full_replay(&shared, &model, &input, generation);
        let snapshot = {
            let mut engine = shared.lock().unwrap();
            if engine.replay_generation != generation {
                return;
            }
            match result {
                Ok((runtime, mut replayed)) => {
                    replayed.signature = input.signature;
                    engine.runtime = Some(runtime);
                    engine.replayed = replayed;
                    engine.scores.clear();
                    engine.answers_since_requery = 0;
                    engine.status = Status::Ready;
                    engine.epoch += 1;
                    engine.snapshot_bytes()
                }
                Err(message) => {
                    engine.fail(message);
                    None
                }
            }
        };
        if let Some((path, bytes)) = snapshot {
            write_snapshot_file(&path, &bytes);
        }
    });
}

/// Restores the snapshot off the collection thread. If it cannot be read, it
/// is removed and the engine goes back to not prepared, so the next
/// preparation replays in full.
fn spawn_restore(shared: SharedEngine, model: Model, path: PathBuf, generation: u64) {
    std::thread::spawn(move || {
        let restored = Engine::read_snapshot(&path, &model);
        let mut engine = shared.lock().unwrap();
        if engine.replay_generation != generation {
            return;
        }
        match restored {
            Some((replayed, runtime)) => {
                engine.runtime = Some(runtime);
                engine.replayed = replayed;
                engine.scores.clear();
                engine.answers_since_requery = 0;
                engine.status = Status::Ready;
            }
            None => {
                let _ = std::fs::remove_file(&path);
                engine.status = Status::NotPrepared;
            }
        }
        engine.epoch += 1;
    });
}

fn full_replay(
    shared: &SharedEngine,
    model: &Model,
    input: &ReplayInput,
    generation: u64,
) -> std::result::Result<(RwkvRuntime, Replayed), String> {
    let rows = history::select_rows(&input.revlog, &input.mappings);
    let total = rows.len() as u64;
    let mut runtime =
        RwkvRuntime::new(model, StatePrecision::Int4, 0).map_err(|e| e.to_string())?;
    let mut replayed = Replayed {
        mappings: input.mappings.clone(),
        rollover_secs: input.timing.next_day_at.0.rem_euclid(86_400),
        ..Default::default()
    };
    let mut inputs: Vec<ReviewInput> = Vec::with_capacity(REPLAY_CHUNK);
    for (index, row) in rows.iter().enumerate() {
        let previous = replayed.cards.get(&row.card_id);
        let (review, card) = history::answered_input(row, previous, index as u64, &input.timing);
        replayed.cards.insert(row.card_id, card);
        inputs.push(review);
        if inputs.len() == REPLAY_CHUNK || index + 1 == rows.len() {
            runtime.replay(&inputs, false).map_err(|e| e.to_string())?;
            inputs.clear();
            let mut engine = shared.lock().unwrap();
            if engine.replay_generation != generation {
                return Err("superseded".into());
            }
            engine.status = Status::Preparing {
                done: index as u64 + 1,
                total,
            };
        }
    }
    replayed.rows = total;
    Ok((runtime, replayed))
}

/// A card answered today comes back only after [MIN_ELAPSED_SECS] and
/// [MIN_INTERVENING_REVIEWS] other answers.
fn same_day_guard_passes(intervening_reviews: u64, elapsed_secs: i64) -> bool {
    intervening_reviews >= MIN_INTERVENING_REVIEWS && elapsed_secs >= MIN_ELAPSED_SECS
}

/// What [Collection::rwkv_catch_up] found.
enum CatchUp {
    Ready,
    NotReady,
}

impl Collection {
    pub fn rwkv_enabled(&self) -> bool {
        self.get_config_optional::<bool, _>(CONFIG_KEY)
            .unwrap_or(false)
    }

    /// True when the queue is built by the RWKV rule right now.
    pub(crate) fn rwkv_in_use(&self) -> bool {
        self.rwkv_enabled()
            && self
                .state
                .rwkv
                .try_lock()
                .is_ok_and(|engine| engine.is_ready())
    }

    fn rwkv_snapshot_path(&self) -> PathBuf {
        self.col_path.with_extension("rwkv")
    }

    pub fn rwkv_status(&self) -> (bool, Status) {
        let status = self.state.rwkv.lock().unwrap().status.clone();
        (self.rwkv_enabled(), status)
    }

    /// Installs the model, then restores the snapshot or starts a full
    /// replay, both in the background. Safe to call again; a ready or
    /// preparing engine is kept.
    pub fn rwkv_prepare(&mut self, model: Model) -> Result<()> {
        let shared = self.state.rwkv.clone();
        // An in-memory collection (tests) has no file to keep a snapshot beside.
        let snapshot_path = self.col_path.is_file().then(|| self.rwkv_snapshot_path());
        let mut engine = shared.lock().unwrap();
        engine.snapshot_path = snapshot_path.clone();
        if engine.is_ready()
            || (matches!(engine.status, Status::Preparing { .. }) && engine.model.is_some())
        {
            return Ok(());
        }
        engine.model = Some(model.clone());
        engine.failed_at = None;
        if let Some(path) = snapshot_path.filter(|path| path.exists()) {
            engine.replay_generation += 1;
            engine.runtime = None;
            engine.status = Status::Preparing { done: 0, total: 0 };
            engine.epoch += 1;
            spawn_restore(shared.clone(), model, path, engine.replay_generation);
            return Ok(());
        }
        let timing = self.timing_today()?;
        self.rwkv_start_full_replay(&shared, &mut engine, &timing)
    }

    fn rwkv_start_full_replay(
        &mut self,
        shared: &SharedEngine,
        engine: &mut Engine,
        timing: &SchedTimingToday,
    ) -> Result<()> {
        let Some(model) = engine.model.clone() else {
            return Ok(());
        };
        let input = ReplayInput {
            revlog: history::read_revlog(self, None)?,
            mappings: Mappings::load(self)?,
            timing: *timing,
            signature: Signature::read(self)?,
        };
        engine.replay_generation += 1;
        engine.runtime = None;
        engine.status = Status::Preparing {
            done: 0,
            total: input.revlog.len() as u64,
        };
        engine.epoch += 1;
        spawn_full_replay(shared.clone(), model, input, engine.replay_generation);
        Ok(())
    }

    /// Brings a ready engine up to date with the collection.
    fn rwkv_catch_up(
        &mut self,
        shared: &SharedEngine,
        engine: &mut Engine,
        timing: &SchedTimingToday,
    ) -> Result<CatchUp> {
        if !engine.is_ready() {
            return Ok(CatchUp::NotReady);
        }
        if timing.next_day_at.0.rem_euclid(86_400) != engine.replayed.rollover_secs {
            self.rwkv_start_full_replay(shared, engine, timing)?;
            return Ok(CatchUp::NotReady);
        }
        let signature = Signature::read(self)?;
        if signature == engine.replayed.signature {
            return Ok(CatchUp::Ready);
        }
        if !self.rwkv_apply_changes(engine, &signature, timing)? {
            self.rwkv_start_full_replay(shared, engine, timing)?;
            return Ok(CatchUp::NotReady);
        }
        engine.replayed.signature = signature;
        Ok(CatchUp::Ready)
    }

    /// Applies changes since the last catch-up incrementally. Returns false
    /// when only a full replay gives the right state.
    fn rwkv_apply_changes(
        &mut self,
        engine: &mut Engine,
        signature: &Signature,
        timing: &SchedTimingToday,
    ) -> Result<bool> {
        let old = engine.replayed.signature;

        // Presets: decks are few, so every catch-up compares them.
        let deck_presets = Mappings::load_deck_presets(self)?;
        engine.replayed.mappings.deck_presets = deck_presets;
        let mappings = &engine.replayed.mappings;
        if engine
            .replayed
            .cards
            .values()
            .any(|h| mappings.preset_for_deck(h.home_deck_id) != h.preset_id)
        {
            return Ok(false);
        }

        // Cards. A sync, a deletion or a count change compares every card;
        // otherwise local changes are found by modification time.
        if signature.last_sync != old.last_sync
            || signature.graves_count != old.graves_count
            || signature.cards_count != old.cards_count
        {
            let cards = Mappings::load_cards(self)?;
            for (card, history) in &engine.replayed.cards {
                match cards.get(card) {
                    Some(&(note, deck))
                        if note == history.note_id && deck == history.home_deck_id => {}
                    _ => return Ok(false),
                }
            }
            engine.replayed.mappings.cards = cards;
        } else if signature.cards_max_mod != old.cards_max_mod {
            let changed: Vec<(i64, i64, i64)> = self
                .storage
                .db
                .prepare(
                    "select id, nid, case when odid != 0 then odid else did end from cards where mod > ?",
                )?
                .query_map([old.cards_max_mod], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?
                .collect::<std::result::Result<_, _>>()?;
            for (card, note, deck) in changed {
                if let Some(history) = engine.replayed.cards.get(&card) {
                    if history.note_id != note || history.home_deck_id != deck {
                        return Ok(false);
                    }
                }
                engine.replayed.mappings.cards.insert(card, (note, deck));
            }
        }

        // Revlog rows removed: only an undo of the newest journal entries can
        // be followed.
        let mut known_count = old.revlog_count;
        let mut known_max = old.revlog_max_id;
        if signature.revlog_max_id < known_max || signature.revlog_count < known_count {
            if engine.replayed.journal.is_empty() {
                return Ok(false);
            }
            let ids = engine
                .replayed
                .journal
                .iter()
                .map(|e| e.review_id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let present: HashSet<i64> = self
                .storage
                .db
                .prepare(&format!("select id from revlog where id in ({ids})"))?
                .query_map([], |row| row.get(0))?
                .collect::<std::result::Result<_, _>>()?;
            while let Some(entry) = engine.replayed.journal.last().cloned() {
                if present.contains(&entry.review_id) {
                    break;
                }
                if !engine.undo_newest(entry.review_id) {
                    return Ok(false);
                }
                known_count -= 1;
            }
            if engine
                .replayed
                .journal
                .iter()
                .any(|e| !present.contains(&e.review_id))
            {
                return Ok(false);
            }
            known_max = self.storage.db.query_row(
                "select coalesce(max(id), 0) from revlog where id <= ?",
                [known_max],
                |row| row.get(0),
            )?;
        }

        // New rows: only rows newer than everything replayed can be appended.
        let new_rows = history::read_revlog(self, Some(known_max))?;
        if known_count + new_rows.len() as u64 != signature.revlog_count {
            // Rows older than the newest replayed one arrived (a sync).
            return Ok(false);
        }
        for row in new_rows {
            if row.is_forget_marker() && engine.replayed.cards.contains_key(&row.card_id) {
                return Ok(false);
            }
            let mapping = engine.replayed.mappings.cards.get(&row.card_id).copied();
            let (Some((note_id, home_deck_id)), true) = (mapping, row.is_replayable()) else {
                engine.push_journal(JournalEntry {
                    review_id: row.id,
                    appended: None,
                });
                continue;
            };
            let previous_kind = engine.replayed.cards.get(&row.card_id).map(|h| h.last_kind);
            // A new learning sequence drops the card's earlier reviews.
            if row.kind == 0 && previous_kind.is_some_and(|kind| kind != 0) {
                return Ok(false);
            }
            let kept = KeptRow {
                review_id: row.id,
                card_id: row.card_id,
                note_id,
                home_deck_id,
                preset_id: engine.replayed.mappings.preset_for_deck(home_deck_id),
                ease: row.ease,
                duration_millis: row.duration_millis,
                review_kind: row.kind,
                is_learning_start: row.kind == 0 && previous_kind.is_none(),
            };
            if !engine.append(&kept, timing) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Runs `func` with the engine when RWKV scheduling is on and the engine
    /// is ready and current. Returns None otherwise, and the caller uses
    /// upstream's rule.
    fn with_ready_rwkv<T>(
        &mut self,
        func: impl FnOnce(&mut Self, &mut Engine, &SchedTimingToday) -> Result<T>,
    ) -> Result<Option<T>> {
        if !self.rwkv_enabled() {
            return Ok(None);
        }
        let shared = self.state.rwkv.clone();
        let needs_preparing = shared.try_lock().is_ok_and(|engine| match engine.status {
            Status::NotPrepared => true,
            Status::Failed(_) => engine
                .failed_at
                .is_none_or(|at| at.elapsed() >= RETRY_FAILED_AFTER),
            _ => false,
        });
        if needs_preparing {
            if let Some(model) = loaded_model() {
                self.rwkv_prepare(model)?;
            }
        }
        // A replay thread holds the lock only briefly; if it is held, the
        // caller falls back rather than wait.
        let Ok(mut engine) = shared.try_lock() else {
            return Ok(None);
        };
        let timing = self.timing_today()?;
        match self.rwkv_catch_up(&shared, &mut engine, &timing) {
            Ok(CatchUp::Ready) => {}
            Ok(CatchUp::NotReady) => return Ok(None),
            Err(err) => {
                engine.fail(err.to_string());
                return Ok(None);
            }
        }
        match func(self, &mut engine, &timing) {
            Ok(value) => Ok(Some(value)),
            Err(err) => {
                engine.fail(err.to_string());
                Ok(None)
            }
        }
    }

    /// The epoch the queues must be built under; differs from the one stored
    /// with the queues after the engine became ready or replayed.
    /// None while a replay thread holds the engine, so callers keep what
    /// they have.
    pub(crate) fn rwkv_epoch(&self) -> Option<u64> {
        if !self.rwkv_enabled() {
            return Some(0);
        }
        self.state.rwkv.try_lock().ok().map(|engine| engine.epoch)
    }

    /// The review cards in their home deck that the RWKV rule queues today,
    /// lowest predicted recall first. `active_only` limits them to the
    /// active decks of the current queue build. None when RWKV is off or not
    /// ready.
    pub(crate) fn rwkv_due_review_cards(
        &mut self,
        active_only: bool,
    ) -> Result<Option<Vec<RwkvDueCard>>> {
        self.with_ready_rwkv(|col, engine, timing| col.rwkv_select(engine, timing, active_only))
    }

    fn rwkv_select(
        &mut self,
        engine: &mut Engine,
        timing: &SchedTimingToday,
        active_only: bool,
    ) -> Result<Vec<RwkvDueCard>> {
        let sql = format!(
            "select id, nid, due, mod, did, odid, reps from cards where queue = {} and odid = 0 {}",
            CardQueue::Review as i8,
            if active_only {
                "and did in (select id from active_decks)"
            } else {
                ""
            }
        );
        let candidates: Vec<DueCard> = self
            .storage
            .db
            .prepare(&sql)?
            .query_map([], |row| {
                Ok(DueCard {
                    id: row.get(0)?,
                    note_id: row.get(1)?,
                    due: row.get(2).ok().unwrap_or_default(),
                    mtime: row.get(3)?,
                    current_deck_id: row.get(4)?,
                    original_deck_id: row.get(5)?,
                    reps: row.get(6)?,
                    kind: DueCardKind::Review,
                })
            })?
            .collect::<std::result::Result<_, _>>()?;

        let decks = self.storage.get_decks_map()?;
        let configs = self.storage.get_deck_config_map()?;
        let target_for = |deck_id: DeckId| -> f32 {
            decks
                .get(&deck_id)
                .and_then(|deck| {
                    let config = configs.get(&deck.config_id()?)?;
                    Some(deck.effective_desired_retention(config))
                })
                .unwrap_or(0.9)
        };

        if engine.answers_since_requery >= REQUERY_EVERY_ANSWERS {
            engine.scores.clear();
            engine.answers_since_requery = 0;
        }
        let today = timing.days_elapsed as i64;
        let day_start_ms = (timing.next_day_at.0 - 86_400) * 1000;
        let end_of_day_ms = timing.next_day_at.0 * 1000 - 1;
        let now_ms = timing.now.0 * 1000;

        let mut due = Vec::new();
        let mut end_of_day_queries: Vec<(DueCard, ReviewInput, CardHistory)> = Vec::new();
        let mut now_queries: Vec<(DueCard, ReviewInput)> = Vec::new();
        for card in candidates {
            let Some(last) = engine.replayed.cards.get(&card.id.0).copied() else {
                // No replayable history: the model cannot score it, so the
                // stored due date decides, as upstream would.
                if card.due <= today as i32 {
                    due.push(RwkvDueCard {
                        card,
                        retrievability: f32::NAN,
                    });
                }
                continue;
            };
            if last.last_review_id >= day_start_ms {
                let intervening = engine.replayed.rows.saturating_sub(last.last_index + 1);
                let elapsed = (now_ms - last.last_review_id) / 1000;
                if !same_day_guard_passes(intervening, elapsed) {
                    continue;
                }
                now_queries.push((card, history::query_input(card.id.0, &last, today, now_ms)));
            } else {
                match engine.scores.get(&card.id.0) {
                    Some(score)
                        if score.day == today && score.last_review_id == last.last_review_id =>
                    {
                        if score.r <= target_for(card.current_deck_id) {
                            due.push(RwkvDueCard {
                                card,
                                retrievability: score.r,
                            });
                        }
                    }
                    _ => end_of_day_queries.push((
                        card,
                        history::query_input(card.id.0, &last, today, end_of_day_ms),
                        last,
                    )),
                }
            }
        }

        let runtime = engine.runtime.as_mut().or_invalid("RWKV runtime missing")?;
        if !end_of_day_queries.is_empty() {
            let inputs: Vec<_> = end_of_day_queries.iter().map(|q| q.1.clone()).collect();
            let scores = runtime.query(&inputs).or_invalid("RWKV query")?;
            for ((card, _, last), r) in end_of_day_queries.into_iter().zip(scores) {
                if !r.is_finite() {
                    continue;
                }
                engine.scores.insert(
                    card.id.0,
                    Score {
                        day: today,
                        last_review_id: last.last_review_id,
                        note_id: last.note_id,
                        r,
                    },
                );
                if r <= target_for(card.current_deck_id) {
                    due.push(RwkvDueCard {
                        card,
                        retrievability: r,
                    });
                }
            }
        }
        if !now_queries.is_empty() {
            let runtime = engine.runtime.as_mut().or_invalid("RWKV runtime missing")?;
            let inputs: Vec<_> = now_queries.iter().map(|q| q.1.clone()).collect();
            let scores = runtime.query(&inputs).or_invalid("RWKV query")?;
            for ((card, _), r) in now_queries.into_iter().zip(scores) {
                if r.is_finite() && r <= target_for(card.current_deck_id) {
                    due.push(RwkvDueCard {
                        card,
                        retrievability: r,
                    });
                }
            }
        }
        // Lowest recall first. Cards the model could not score sort last.
        due.sort_by(|a, b| {
            let key = |c: &RwkvDueCard| {
                if c.retrievability.is_finite() {
                    c.retrievability
                } else {
                    f32::INFINITY
                }
            };
            key(a).total_cmp(&key(b)).then(a.card.id.cmp(&b.card.id))
        });
        Ok(due)
    }

    /// Review counts per home deck under the RWKV rule, for the deck list.
    pub(crate) fn rwkv_review_counts_by_deck(&mut self) -> Result<Option<HashMap<DeckId, u32>>> {
        Ok(self.rwkv_due_review_cards(false)?.map(|cards| {
            let mut counts = HashMap::new();
            for c in cards {
                *counts.entry(c.card.current_deck_id).or_insert(0) += 1;
            }
            counts
        }))
    }
}

#[cfg(test)]
mod test;
