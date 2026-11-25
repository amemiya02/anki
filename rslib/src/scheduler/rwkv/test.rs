// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Tests that need a model read its weights from the folder named by
//! HAYAKI_RWKV_WEIGHTS_DIR, under the file names of [ModelSize::weights], and
//! are skipped when it is unset, so the suite runs without network access.

use std::time::Duration;
use std::time::Instant;

use super::*;
use crate::card::CardType;
use crate::scheduler::queue::QueueEntryKind;

fn weights_of(size: ModelSize) -> Option<LoadedModel> {
    let dir = std::env::var("HAYAKI_RWKV_WEIGHTS_DIR").ok()?;
    let bytes = std::fs::read(Path::new(&dir).join(size.weights().file_name)).unwrap();
    assert!(size.weights().is_valid(&bytes));
    Some(LoadedModel::from_bytes(size, &bytes).unwrap())
}

fn weights() -> Option<LoadedModel> {
    weights_of(ModelSize::Small)
}

const CARDS: i64 = 40;

/// A collection whose cards are all review cards with two months of history.
/// Some cards are due by their stored date and some are not.
fn collection_with_history() -> Collection {
    let mut col = Collection::new();
    let nt = col.get_notetype_by_name("Basic").unwrap().unwrap();
    let deck = col.get_or_create_normal_deck("Default").unwrap();
    let now = TimestampSecs::now().0;
    let mut cards = vec![];
    for i in 0..CARDS {
        let mut note = nt.new_note();
        note.set_field(0, format!("front {i}")).unwrap();
        col.add_note(&mut note, deck.id).unwrap();
        let mut card = col
            .storage
            .get_card_by_ordinal(note.id, 0)
            .unwrap()
            .unwrap();
        let days = [60, 59 - i % 5, 50 - i % 9, 30 - i % 11, 12 - i % 7];
        for (n, days_ago) in days.iter().enumerate() {
            let id = (now - days_ago * 86_400 - 3_600) * 1000 + i;
            let (kind, ease) = if n == 0 {
                (0, 3)
            } else {
                (1, if (i + n as i64) % 6 == 0 { 1 } else { 3 })
            };
            col.storage
                .db
                .execute(
                    "insert into revlog (id, cid, usn, ease, ivl, lastIvl, factor, time, type) values (?, ?, 0, ?, 5, 3, 2500, 4000, ?)",
                    [id, card.id.0, ease, kind],
                )
                .unwrap();
        }
        let today = col.timing_today().unwrap().days_elapsed as i32;
        card.ctype = CardType::Review;
        card.queue = CardQueue::Review;
        card.interval = 10;
        card.due = today + (i as i32 % 9) - 4;
        cards.push(card);
    }
    col.update_cards_maybe_undoable(cards, false).unwrap();
    col
}

fn review_queue(col: &mut Collection) -> Vec<CardId> {
    col.clear_study_queues();
    col.get_queued_cards(1000, false)
        .unwrap()
        .cards
        .into_iter()
        .filter(|c| c.kind == QueueEntryKind::Review)
        .map(|c| c.card.id)
        .collect()
}

fn set_switch(col: &mut Collection, on: bool) {
    col.set_config_json(CONFIG_KEY, &on, false).unwrap();
}

fn wait_ready(col: &Collection) {
    let started = Instant::now();
    loop {
        match col.rwkv_status().2 {
            Status::Ready => return,
            Status::Failed(err) => panic!("RWKV failed: {err}"),
            _ if started.elapsed() > Duration::from_secs(60) => panic!("RWKV not ready"),
            _ => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

fn prepared(col: &mut Collection) -> bool {
    let Some(model) = weights() else {
        println!("HAYAKI_RWKV_WEIGHTS_DIR not set; skipping");
        return false;
    };
    set_switch(col, true);
    col.rwkv_prepare(model).unwrap();
    wait_ready(col);
    true
}

fn scores(col: &mut Collection) -> Vec<(CardId, u32)> {
    col.rwkv_due_review_cards(false)
        .unwrap()
        .unwrap()
        .into_iter()
        .map(|c| (c.card.id, c.retrievability.to_bits()))
        .collect()
}

#[test]
fn switch_on_without_a_ready_model_keeps_the_upstream_queue() {
    let mut col = collection_with_history();
    let upstream = review_queue(&mut col);
    assert!(!upstream.is_empty());
    set_switch(&mut col, true);
    assert_eq!(col.rwkv_status().2, Status::NotPrepared);
    assert_eq!(review_queue(&mut col), upstream);
    set_switch(&mut col, false);
    assert_eq!(review_queue(&mut col), upstream);
}

#[test]
fn queue_follows_recall_and_deck_counts_agree() {
    let mut col = collection_with_history();
    let upstream = review_queue(&mut col);
    if !prepared(&mut col) {
        return;
    }
    let due = col.rwkv_due_review_cards(false).unwrap().unwrap();
    let expected: Vec<CardId> = due.iter().map(|c| c.card.id).collect();
    assert!(due
        .windows(2)
        .all(|w| w[0].retrievability <= w[1].retrievability));
    assert!(due.iter().all(|c| c.retrievability <= 0.9));
    let queue = review_queue(&mut col);
    assert_eq!(queue, expected);
    assert_ne!(
        queue, upstream,
        "the RWKV rule should differ from due dates here"
    );

    let tree = col.deck_tree(Some(TimestampSecs::now())).unwrap();
    let default = tree.children.iter().find(|d| d.name == "Default").unwrap();
    assert_eq!(default.review_count as usize, queue.len());

    // Turning the switch off returns to the stored due dates at once.
    set_switch(&mut col, false);
    assert_eq!(review_queue(&mut col), upstream);
}

#[test]
fn undo_restores_every_prediction() {
    let mut col = collection_with_history();
    if !prepared(&mut col) {
        return;
    }
    let before = scores(&mut col);
    let rows_before = col.state.rwkv.lock().unwrap().replayed.rows;
    let answered = col.answer_good().card_id;
    let after_answer = scores(&mut col);
    assert_eq!(
        col.state.rwkv.lock().unwrap().replayed.rows,
        rows_before + 1
    );
    assert_ne!(before, after_answer);
    // The answered card waits for the same-day guard.
    assert!(!review_queue(&mut col).contains(&answered));

    col.undo().unwrap();
    let after_undo = scores(&mut col);
    assert_eq!(col.state.rwkv.lock().unwrap().replayed.rows, rows_before);
    assert_eq!(before, after_undo);

    // Redo appends the same review again and gives the same predictions.
    col.redo().unwrap();
    assert_eq!(scores(&mut col), after_answer);
}

#[test]
fn history_changes_force_a_full_replay() {
    let mut col = collection_with_history();
    if !prepared(&mut col) {
        return;
    }
    let first = col.storage.get_all_cards().into_iter().next().unwrap();
    let other = col.get_or_create_normal_deck("Other").unwrap();
    col.set_deck(&[first.id], other.id).unwrap();
    // The move changes a replayed card's home deck, so the engine replays.
    assert!(col.rwkv_due_review_cards(false).unwrap().is_none());
    wait_ready(&col);
    assert!(col.rwkv_due_review_cards(false).unwrap().is_some());
}

#[test]
fn a_reopened_collection_restores_the_snapshot() {
    let Some(model) = weights() else {
        println!("HAYAKI_RWKV_WEIGHTS_DIR not set; skipping");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("collection.anki2");
    let expected = {
        // Write the in-memory fixture to a file-backed collection.
        collection_with_history()
            .storage
            .db
            .execute(&format!("vacuum into '{}'", path.display()), [])
            .unwrap();
        let mut col = crate::collection::CollectionBuilder::new(&path)
            .build()
            .unwrap();
        set_switch(&mut col, true);
        col.rwkv_prepare(model.clone()).unwrap();
        wait_ready(&col);
        let expected = scores(&mut col);
        col.close(None).unwrap();
        expected
    };
    assert!(path.with_extension("rwkv").exists());
    let mut col = crate::collection::CollectionBuilder::new(&path)
        .build()
        .unwrap();
    col.rwkv_prepare(model).unwrap();
    // Restoring a snapshot reports no replay progress.
    assert_eq!(col.rwkv_status().2, Status::Preparing { done: 0, total: 0 });
    wait_ready(&col);
    assert_eq!(scores(&mut col), expected);
}

fn is_replaying(col: &Collection) -> bool {
    matches!(col.rwkv_status().2, Status::Preparing { total, .. } if total > 0)
}

/// A sync can bring a card moved on another device with a modification time
/// older than local ones; the changed sync time still finds it.
#[test]
fn a_synced_move_with_an_old_modification_time_replays_in_full() {
    let mut col = collection_with_history();
    if !prepared(&mut col) {
        return;
    }
    let other = col.get_or_create_normal_deck("Other").unwrap();
    let card = col
        .storage
        .get_all_cards()
        .into_iter()
        .min_by_key(|c| c.id)
        .unwrap();
    col.storage
        .db
        .execute(
            "update cards set did = ?, mod = 1 where id = ?",
            [other.id.0, card.id.0],
        )
        .unwrap();
    col.storage.set_last_sync(TimestampMillis::now()).unwrap();
    assert!(col.rwkv_due_review_cards(false).unwrap().is_none());
    assert!(is_replaying(&col));
    wait_ready(&col);
}

/// A deletion and an addition that leave the card count unchanged are still
/// found, through the deletion's grave.
#[test]
fn a_deletion_hidden_by_an_addition_replays_in_full() {
    let mut col = collection_with_history();
    if !prepared(&mut col) {
        return;
    }
    let card = col
        .storage
        .get_all_cards()
        .into_iter()
        .min_by_key(|c| c.id)
        .unwrap();
    col.remove_notes(&[card.note_id]).unwrap();
    let nt = col.get_notetype_by_name("Basic").unwrap().unwrap();
    let mut note = nt.new_note();
    note.set_field(0, "replacement").unwrap();
    col.add_note(&mut note, DeckId(1)).unwrap();
    assert!(col.rwkv_due_review_cards(false).unwrap().is_none());
    assert!(is_replaying(&col));
    wait_ready(&col);
}

/// Undoing a change the model does not replay, such as Set Due Date, is
/// followed without a full replay.
#[test]
fn undoing_a_set_due_date_needs_no_replay() {
    let mut col = collection_with_history();
    if !prepared(&mut col) {
        return;
    }
    let before = scores(&mut col);
    let card = col
        .storage
        .get_all_cards()
        .into_iter()
        .min_by_key(|c| c.id)
        .unwrap();
    col.set_due_date(&[card.id], "5", None).unwrap();
    assert!(col.rwkv_due_review_cards(false).unwrap().is_some());
    col.undo().unwrap();
    assert!(col.rwkv_due_review_cards(false).unwrap().is_some());
    assert!(!is_replaying(&col));
    assert_eq!(scores(&mut col), before);
}

/// Answering writes the same card and revlog with the switch on as off, so
/// unpatched Anki always sees an ordinary collection.
#[test]
fn answering_writes_the_same_rows_with_the_switch_on() {
    let Some(_) = weights() else {
        println!("HAYAKI_RWKV_WEIGHTS_DIR not set; skipping");
        return;
    };
    let answer_first_card = |col: &mut Collection| {
        let card = col
            .storage
            .get_all_cards()
            .into_iter()
            .min_by_key(|c| c.id)
            .unwrap();
        let states = col.get_scheduling_states(card.id).unwrap();
        col.answer_card(&mut crate::scheduler::answering::CardAnswer {
            card_id: card.id,
            current_state: states.current,
            new_state: states.good,
            rating: crate::scheduler::answering::Rating::Good,
            answered_at: TimestampMillis(1_000),
            milliseconds_taken: 3_000,
            custom_data: None,
            from_queue: true,
        })
        .unwrap();
        let card = col.storage.get_card(card.id).unwrap().unwrap();
        let revlog = col.storage.get_revlog_entries_for_card(card.id).unwrap();
        let last = revlog.into_iter().max_by_key(|r| r.id).unwrap();
        (
            (
                card.due,
                card.interval,
                card.ease_factor,
                card.reps,
                card.lapses,
                card.queue,
                card.ctype,
            ),
            card.memory_state
                .map(|m| (m.stability.to_bits(), m.difficulty.to_bits())),
            (
                last.button_chosen,
                last.interval,
                last.last_interval,
                last.ease_factor,
                last.review_kind,
            ),
        )
    };

    let mut off = collection_with_history();
    let mut on = collection_with_history();
    assert!(prepared(&mut on));
    // Build the RWKV queue first, so answering runs with it in place.
    let _ = review_queue(&mut on);
    assert_eq!(answer_first_card(&mut on), answer_first_card(&mut off));
}

#[test]
fn the_same_day_guard_needs_both_time_and_other_answers() {
    assert!(!same_day_guard_passes(4, 600));
    assert!(!same_day_guard_passes(50, 29));
    assert!(same_day_guard_passes(5, 30));
}

/// Filtered decks, new cards and learning cards keep upstream's rule.
#[test]
fn only_home_deck_reviews_follow_the_rwkv_rule() {
    let mut col = collection_with_history();
    // A new card, and a filtered deck holding some of the review cards.
    let nt = col.get_notetype_by_name("Basic").unwrap().unwrap();
    let mut note = nt.new_note();
    note.set_field(0, "new").unwrap();
    col.add_note(&mut note, DeckId(1)).unwrap();
    let mut filtered = col.get_or_create_filtered_deck(DeckId(0)).unwrap();
    filtered.human_name = "Filtered".into();
    filtered.config.search_terms[0].search = "deck:Default is:review".into();
    filtered.config.search_terms[0].limit = 20;
    col.add_or_update_filtered_deck(filtered).unwrap();
    let filtered_id = col.get_deck_id("Filtered").unwrap().unwrap();

    let queue_of = |col: &mut Collection, deck: DeckId, kind: QueueEntryKind| -> Vec<CardId> {
        col.set_current_deck(deck).unwrap();
        col.clear_study_queues();
        col.get_queued_cards(1000, false)
            .unwrap()
            .cards
            .into_iter()
            .filter(|c| c.kind == kind)
            .map(|c| c.card.id)
            .collect()
    };
    let filtered_off = queue_of(&mut col, filtered_id, QueueEntryKind::Review);
    let new_off = queue_of(&mut col, DeckId(1), QueueEntryKind::New);
    if !prepared(&mut col) {
        return;
    }
    assert!(!filtered_off.is_empty());
    assert_eq!(
        queue_of(&mut col, filtered_id, QueueEntryKind::Review),
        filtered_off
    );
    assert_eq!(queue_of(&mut col, DeckId(1), QueueEntryKind::New), new_off);
}

#[test]
fn changing_the_day_rollover_replays_in_full() {
    let mut col = collection_with_history();
    if !prepared(&mut col) {
        return;
    }
    let mut prefs = col.get_preferences().unwrap();
    let scheduling = prefs.scheduling.as_mut().unwrap();
    scheduling.rollover = (scheduling.rollover + 1) % 24;
    col.set_preferences(prefs).unwrap();
    assert!(col.rwkv_due_review_cards(false).unwrap().is_none());
    wait_ready(&col);
    assert!(col.rwkv_due_review_cards(false).unwrap().is_some());
}

/// Undo writes a card back with its old modification time; moving a card
/// and undoing the move still leaves the model with the card's real deck.
#[test]
fn undoing_a_deck_move_is_noticed() {
    let mut col = collection_with_history();
    if !prepared(&mut col) {
        return;
    }
    let other = col.get_or_create_normal_deck("Other").unwrap();
    let card = col
        .storage
        .get_all_cards()
        .into_iter()
        .min_by_key(|c| c.id)
        .unwrap();
    col.set_deck(&[card.id], other.id).unwrap();
    assert!(col.rwkv_due_review_cards(false).unwrap().is_none());
    wait_ready(&col);
    assert!(col.rwkv_due_review_cards(false).unwrap().is_some());
    col.undo().unwrap();
    assert!(col.rwkv_due_review_cards(false).unwrap().is_none());
    assert!(is_replaying(&col));
    wait_ready(&col);
    let home = col.state.rwkv.lock().unwrap().replayed.cards[&card.id.0].home_deck_id;
    assert_eq!(home, card.deck_id.0);
}

fn set_size(col: &mut Collection, size: ModelSize) {
    col.rwkv_set_size(size).unwrap();
}

#[test]
fn the_model_choice_defaults_to_small() {
    let mut col = collection_with_history();
    assert_eq!(col.rwkv_size(), ModelSize::Small);
    set_size(&mut col, ModelSize::Tiny);
    assert_eq!(col.rwkv_size(), ModelSize::Tiny);
    // An unknown value, say from a newer app, falls back to small.
    col.set_config_json(SIZE_CONFIG_KEY, &"huge", false)
        .unwrap();
    assert_eq!(col.rwkv_size(), ModelSize::Small);
}

#[test]
fn every_size_prepares_and_queues() {
    for size in ModelSize::ALL {
        let mut col = collection_with_history();
        let Some(model) = weights_of(size) else {
            println!("HAYAKI_RWKV_WEIGHTS_DIR not set; skipping");
            return;
        };
        set_switch(&mut col, true);
        set_size(&mut col, size);
        col.rwkv_prepare(model).unwrap();
        wait_ready(&col);
        let due = col.rwkv_due_review_cards(false).unwrap().unwrap();
        assert!(due.iter().all(|c| c.retrievability <= 0.9), "{size:?}");
        assert_eq!(
            review_queue(&mut col),
            due.iter().map(|c| c.card.id).collect::<Vec<_>>(),
            "{size:?}"
        );
    }
}

#[test]
fn choosing_another_model_drops_the_old_one_until_the_new_one_is_prepared() {
    let mut col = collection_with_history();
    let upstream = review_queue(&mut col);
    if !prepared(&mut col) {
        return;
    }
    let small = scores(&mut col);
    set_size(&mut col, ModelSize::Tiny);
    assert_eq!(col.rwkv_status().2, Status::NotPrepared);
    // No tiny model is loaded, so study falls back.
    assert!(col.rwkv_due_review_cards(false).unwrap().is_none());
    assert_eq!(review_queue(&mut col), upstream);

    // The small model no longer prepares the collection.
    col.rwkv_prepare(weights().unwrap()).unwrap();
    assert_eq!(col.rwkv_status().2, Status::NotPrepared);

    let Some(tiny) = weights_of(ModelSize::Tiny) else {
        return;
    };
    col.rwkv_prepare(tiny).unwrap();
    wait_ready(&col);
    assert_ne!(scores(&mut col), small);

    // Back to small: a full replay gives the same predictions as before.
    set_size(&mut col, ModelSize::Small);
    col.rwkv_prepare(weights().unwrap()).unwrap();
    wait_ready(&col);
    assert_eq!(scores(&mut col), small);
}

#[test]
fn a_synced_choice_is_followed_at_the_next_study() {
    let mut col = collection_with_history();
    let upstream = review_queue(&mut col);
    if !prepared(&mut col) {
        return;
    }
    // A sync writes the key without going through rwkv_set_size.
    col.set_config_json(SIZE_CONFIG_KEY, &ModelSize::Big, false)
        .unwrap();
    assert_eq!(col.rwkv_status().2, Status::NotPrepared);
    assert_eq!(review_queue(&mut col), upstream);
}

/// The process keeps the model the app last loaded, so a collection prepares
/// itself at study time, but only from a model of the size it chose.
#[test]
fn study_prepares_from_the_loaded_model_of_the_chosen_size() {
    let (Some(small), Some(tiny)) = (weights(), weights_of(ModelSize::Tiny)) else {
        println!("HAYAKI_RWKV_WEIGHTS_DIR not set; skipping");
        return;
    };
    // The slot is process-wide; other tests never set it.
    let previous = LOADED_MODEL.lock().unwrap().take();
    let mut col = collection_with_history();
    set_switch(&mut col, true);
    set_size(&mut col, ModelSize::Tiny);
    set_loaded_model(small);
    assert!(col.rwkv_due_review_cards(false).unwrap().is_none());
    assert_eq!(col.rwkv_status().2, Status::NotPrepared);

    set_loaded_model(tiny);
    assert!(col.rwkv_due_review_cards(false).unwrap().is_none());
    wait_ready(&col);
    assert!(col.rwkv_due_review_cards(false).unwrap().is_some());
    *LOADED_MODEL.lock().unwrap() = previous;
}

#[test]
fn choosing_a_model_during_a_replay_drops_that_replay() {
    let mut col = collection_with_history();
    let (Some(small), Some(tiny)) = (weights(), weights_of(ModelSize::Tiny)) else {
        println!("HAYAKI_RWKV_WEIGHTS_DIR not set; skipping");
        return;
    };
    set_switch(&mut col, true);
    col.rwkv_prepare(small).unwrap();
    set_size(&mut col, ModelSize::Tiny);
    assert_eq!(col.rwkv_status().2, Status::NotPrepared);
    col.rwkv_prepare(tiny).unwrap();
    wait_ready(&col);
    // The superseded small replay never installs its runtime.
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(col.rwkv_status().2, Status::Ready);
    assert_eq!(
        col.state.rwkv.lock().unwrap().size(),
        Some(ModelSize::Tiny)
    );
}

#[test]
fn a_snapshot_of_another_model_is_not_restored() {
    let (Some(small), Some(tiny)) = (weights(), weights_of(ModelSize::Tiny)) else {
        println!("HAYAKI_RWKV_WEIGHTS_DIR not set; skipping");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("collection.anki2");
    collection_with_history()
        .storage
        .db
        .execute(&format!("vacuum into '{}'", path.display()), [])
        .unwrap();
    {
        let mut col = crate::collection::CollectionBuilder::new(&path)
            .build()
            .unwrap();
        set_switch(&mut col, true);
        col.rwkv_prepare(small).unwrap();
        wait_ready(&col);
        set_size(&mut col, ModelSize::Tiny);
        col.close(None).unwrap();
    }
    let mut col = crate::collection::CollectionBuilder::new(&path)
        .build()
        .unwrap();
    col.rwkv_prepare(tiny).unwrap();
    wait_ready(&col);
    assert!(col.rwkv_due_review_cards(false).unwrap().is_some());
    // The tiny model's state replaced the small model's snapshot on disk.
    let bytes = std::fs::read(path.with_extension("rwkv")).unwrap();
    let sha = ModelSize::Tiny.weights().sha256.as_bytes();
    assert_eq!(
        &bytes[SNAPSHOT_MAGIC.len()..SNAPSHOT_MAGIC.len() + sha.len()],
        sha
    );
}
