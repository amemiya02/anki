// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Hayaki: turns the revlog into the rows the RWKV model replays.
//!
//! The selection rules are ported from kuma3-anki-core
//! (`rslib/src/storage/revlog/mod.rs`, `rwkv_historical_review_rows_with_cutoffs`
//! and `retain_rwkv_rows_from_latest_learning_start`, commit c639bcec), and the
//! field mapping from its `scheduler/rwkv.rs` (`walk`) and
//! `scheduler/rwkv/history.rs` (`historical_review`).

use std::collections::HashMap;

use fushi_rwkv::ReviewInput;
use serde::Deserialize;
use serde::Serialize;

use crate::prelude::*;
use crate::scheduler::timing::SchedTimingToday;

/// One revlog row that survived the selection rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KeptRow {
    pub review_id: i64,
    pub card_id: i64,
    pub note_id: i64,
    pub home_deck_id: i64,
    pub preset_id: i64,
    pub ease: i64,
    pub duration_millis: i64,
    pub review_kind: i64,
    pub is_learning_start: bool,
}

/// What the replay remembers about a card, so later rows and queries can be
/// built without reading the card's history again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CardHistory {
    pub last_review_id: i64,
    pub last_day: i64,
    pub last_kind: i64,
    /// Index of the card's last kept row in the whole replayed timeline.
    pub last_index: u64,
    pub note_id: i64,
    pub home_deck_id: i64,
    pub preset_id: i64,
}

/// Which cards exist, with their note and home deck, and which preset each
/// normal deck uses. A change to any of these for a replayed card changes the
/// replayed features, so it forces a full replay.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct Mappings {
    pub cards: HashMap<i64, (i64, i64)>,
    pub deck_presets: HashMap<i64, i64>,
}

impl Mappings {
    pub fn load(col: &Collection) -> Result<Self> {
        Ok(Self {
            cards: Self::load_cards(col)?,
            deck_presets: Self::load_deck_presets(col)?,
        })
    }

    pub fn load_cards(col: &Collection) -> Result<HashMap<i64, (i64, i64)>> {
        col.storage
            .db
            .prepare("select id, nid, case when odid != 0 then odid else did end from cards")?
            .query_map([], |row| Ok((row.get(0)?, (row.get(1)?, row.get(2)?))))?
            .collect::<std::result::Result<_, _>>()
            .map_err(Into::into)
    }

    pub fn load_deck_presets(col: &Collection) -> Result<HashMap<i64, i64>> {
        Ok(col
            .storage
            .get_all_decks()?
            .into_iter()
            .filter_map(|deck| Some((deck.id.0, deck.config_id()?.0)))
            .collect())
    }

    pub fn preset_for_deck(&self, deck_id: i64) -> i64 {
        // A missing deck or a filtered home deck has no preset. Fall back to
        // the default preset id, as the kuma3 replay does for unknown decks.
        self.deck_presets.get(&deck_id).copied().unwrap_or(1)
    }
}

/// The scheduling day of a review, counted the way kuma3 counts it: the
/// collection's days_elapsed minus whole days before today's rollover.
pub(crate) fn day_offset(review_id: i64, timing: &SchedTimingToday) -> i64 {
    let review_secs = review_id / 1000;
    let days_before_today = (timing.next_day_at.0 - 1 - review_secs).max(0) / 86_400;
    (timing.days_elapsed as i64 - days_before_today).max(0)
}

/// A raw revlog row before selection.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RevlogRow {
    pub id: i64,
    pub card_id: i64,
    pub ease: i64,
    pub duration_millis: i64,
    pub kind: i64,
    pub factor: i64,
}

impl RevlogRow {
    /// A Forget writes a manual row with ease 0 and factor 0.
    pub fn is_forget_marker(&self) -> bool {
        self.kind == 4 && self.ease == 0 && self.factor == 0
    }

    /// Answers the model replays: rated, a known kind, and not a filtered
    /// review that left scheduling unchanged.
    pub fn is_replayable(&self) -> bool {
        (1..=4).contains(&self.ease)
            && (0..=5).contains(&self.kind)
            && !(self.kind == 3 && self.factor == 0)
    }
}

pub(crate) fn read_revlog(col: &Collection, after_id: Option<i64>) -> Result<Vec<RevlogRow>> {
    let sql = match after_id {
        Some(_) => "select id, cid, ease, time, type, cast(factor as integer) from revlog where id > ? order by id",
        None => "select id, cid, ease, time, type, cast(factor as integer) from revlog where ? order by id",
    };
    let arg = after_id.unwrap_or(1);
    col.storage
        .db
        .prepare(sql)?
        .query_map([arg], |row| {
            Ok(RevlogRow {
                id: row.get(0)?,
                card_id: row.get(1)?,
                ease: row.get(2)?,
                duration_millis: row.get(3)?,
                kind: row.get(4)?,
                factor: row.get(5)?,
            })
        })?
        .collect::<std::result::Result<_, _>>()
        .map_err(Into::into)
}

/// Applies the selection rules to the whole revlog.
pub(crate) fn select_rows(revlog: &[RevlogRow], mappings: &Mappings) -> Vec<KeptRow> {
    let mut latest_forget: HashMap<i64, i64> = HashMap::new();
    for row in revlog {
        if row.is_forget_marker() {
            latest_forget.insert(row.card_id, row.id);
        }
    }
    let mut rows: Vec<KeptRow> = revlog
        .iter()
        .filter(|row| row.is_replayable())
        .filter(|row| {
            latest_forget
                .get(&row.card_id)
                .is_none_or(|&forget| row.id > forget)
        })
        .filter_map(|row| {
            let &(note_id, home_deck_id) = mappings.cards.get(&row.card_id)?;
            Some(KeptRow {
                review_id: row.id,
                card_id: row.card_id,
                note_id,
                home_deck_id,
                preset_id: mappings.preset_for_deck(home_deck_id),
                ease: row.ease,
                duration_millis: row.duration_millis,
                review_kind: row.kind,
                is_learning_start: false,
            })
        })
        .collect();
    retain_from_latest_learning_start(&mut rows);
    rows
}

/// Drops each card's reviews before its latest learning start: a learning
/// review not preceded by another learning review. Cards without one keep
/// their full history.
fn retain_from_latest_learning_start(rows: &mut Vec<KeptRow>) {
    struct Start {
        previous_kind: i64,
        latest_learning_start: Option<i64>,
        first_review_id: i64,
    }
    let mut starts: HashMap<i64, Start> = HashMap::new();
    for row in rows.iter() {
        let learning = row.review_kind == 0;
        starts
            .entry(row.card_id)
            .and_modify(|start| {
                if learning && start.previous_kind != 0 {
                    start.latest_learning_start = Some(row.review_id);
                }
                start.previous_kind = row.review_kind;
            })
            .or_insert(Start {
                previous_kind: row.review_kind,
                latest_learning_start: learning.then_some(row.review_id),
                first_review_id: row.review_id,
            });
    }
    rows.retain_mut(|row| {
        let start = &starts[&row.card_id];
        let start_id = start.latest_learning_start.unwrap_or(start.first_review_id);
        row.is_learning_start = Some(row.review_id) == start.latest_learning_start;
        row.review_id >= start_id
    });
}

/// Builds the answered input for `row`, given what is known about the card's
/// previous kept review, and returns the card's updated history.
pub(crate) fn answered_input(
    row: &KeptRow,
    previous: Option<&CardHistory>,
    index: u64,
    timing: &SchedTimingToday,
) -> (ReviewInput, CardHistory) {
    let day = day_offset(row.review_id, timing);
    // The first kept review has no previous review. kuma3 can measure a
    // learning start from card creation, but the model ignores elapsed time on
    // an answered learning start, so -1 gives the same features.
    let (elapsed_days, elapsed_seconds) = match previous {
        Some(prev) => (
            (day - prev.last_day).max(0),
            ((row.review_id - prev.last_review_id) / 1000).max(0),
        ),
        None => (-1, -1),
    };
    let input = ReviewInput {
        card_id: row.card_id,
        note_id: Some(row.note_id),
        deck_id: Some(row.home_deck_id),
        preset_id: Some(row.preset_id),
        is_query: false,
        ease: Some(row.ease as u8),
        duration_millis: Some(row.duration_millis),
        card_type: Some(if row.is_learning_start {
            0
        } else {
            row.review_kind + 1
        }),
        day_offset: Some(day),
        current_elapsed_days: Some(elapsed_days),
        current_elapsed_seconds: Some(elapsed_seconds),
    };
    let history = CardHistory {
        last_review_id: row.review_id,
        last_day: day,
        last_kind: row.review_kind,
        last_index: index,
        note_id: row.note_id,
        home_deck_id: row.home_deck_id,
        preset_id: row.preset_id,
    };
    (input, history)
}

/// A query for a review card that was last answered at `last`: what is the
/// chance of recall at `at_ms`, counted on scheduling day `day`.
pub(crate) fn query_input(card_id: i64, last: &CardHistory, day: i64, at_ms: i64) -> ReviewInput {
    ReviewInput {
        card_id,
        note_id: Some(last.note_id),
        deck_id: Some(last.home_deck_id),
        preset_id: Some(last.preset_id),
        is_query: true,
        ease: None,
        duration_millis: None,
        card_type: Some(2),
        day_offset: Some(day),
        current_elapsed_days: Some((day - last.last_day).max(0)),
        current_elapsed_seconds: Some(((at_ms - last.last_review_id) / 1000).max(0)),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn kept(id: i64, card: i64, kind: i64) -> KeptRow {
        KeptRow {
            review_id: id,
            card_id: card,
            note_id: card,
            home_deck_id: 1,
            preset_id: 1,
            ease: 3,
            duration_millis: 1000,
            review_kind: kind,
            is_learning_start: false,
        }
    }

    #[test]
    fn keeps_only_the_latest_learning_sequence() {
        let mut rows = vec![
            kept(1, 10, 0),
            kept(2, 10, 1),
            kept(3, 10, 0),
            kept(4, 10, 0),
            kept(5, 10, 1),
            kept(6, 11, 1),
        ];
        retain_from_latest_learning_start(&mut rows);
        let ids: Vec<_> = rows.iter().map(|r| r.review_id).collect();
        assert_eq!(ids, vec![3, 4, 5, 6]);
        assert!(rows[0].is_learning_start);
        assert!(!rows[1].is_learning_start);
        assert!(!rows[3].is_learning_start);
    }

    #[test]
    fn drops_reviews_before_a_forget_and_unrated_rows() {
        let row = |id, ease, kind, factor| RevlogRow {
            id,
            card_id: 10,
            ease,
            duration_millis: 0,
            kind,
            factor,
        };
        let revlog = vec![
            row(1, 3, 0, 0),
            row(2, 3, 1, 2500),
            row(3, 0, 4, 0),
            row(4, 3, 0, 0),
            row(5, 3, 3, 0),
            row(6, 0, 5, 2500),
        ];
        let mappings = Mappings {
            cards: HashMap::from([(10, (100, 1))]),
            deck_presets: HashMap::from([(1, 7)]),
        };
        let rows = select_rows(&revlog, &mappings);
        let ids: Vec<_> = rows.iter().map(|r| r.review_id).collect();
        assert_eq!(ids, vec![4]);
        assert_eq!(rows[0].preset_id, 7);
        assert!(rows[0].is_learning_start);
    }

    #[test]
    fn skips_reviews_of_deleted_cards() {
        let revlog = vec![RevlogRow {
            id: 1,
            card_id: 99,
            ease: 3,
            duration_millis: 0,
            kind: 1,
            factor: 2500,
        }];
        assert!(select_rows(&revlog, &Mappings::default()).is_empty());
    }
}
