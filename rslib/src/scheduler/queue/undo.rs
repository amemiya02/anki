// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

use super::CardQueues;
use super::LearningQueueEntry;
use super::QueueEntry;
use crate::prelude::*;

#[derive(Debug)]
pub(crate) enum UndoableQueueChange {
    CardAnswered(Box<QueueUpdate>),
    CardAnswerUndone(Box<QueueUpdate>),
    /// Hayaki: an answer from queues the RWKV rule built, which dropped them.
    RwkvCardAnswered(RwkvAnswer),
    RwkvCardAnswerUndone(RwkvAnswer),
}

/// Hayaki: the card answered from queues the RWKV rule built, and their day
/// and learning cutoff then. Undoing the answer rebuilds the queues with the
/// card on top under that cutoff, as upstream's undo restores its queues.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RwkvAnswer {
    pub card_id: CardId,
    pub day: u32,
    pub learning_cutoff: TimestampSecs,
}

#[derive(Debug)]
pub(crate) struct QueueUpdate {
    pub entry: QueueEntry,
    pub learning_requeue: Option<LearningQueueEntry>,
    pub queue_build_time: TimestampMillis,
    pub cutoff_snapshot: CutoffSnapshot,
}

/// Stores the old learning count and cutoff prior to the
/// cutoff being adjusted after answering a card.
#[derive(Debug)]
pub(crate) struct CutoffSnapshot {
    pub learning_count: usize,
    pub learning_cutoff: TimestampSecs,
}

impl Collection {
    pub(crate) fn undo_queue_change(&mut self, change: UndoableQueueChange) -> Result<()> {
        match change {
            UndoableQueueChange::CardAnswered(update) => {
                if let Some(queues) = self.get_or_invalidate_queues(update.queue_build_time)? {
                    queues.restore_cutoff(&update.cutoff_snapshot);
                    if let Some(learning) = &update.learning_requeue {
                        queues.remove_intraday_learning_card(learning.id);
                    }
                    queues.push_undo_entry(update.entry);
                }

                if let Some(card_queues) = self.state.card_queues.as_mut() {
                    if let Some(load_balancer) = card_queues.load_balancer.as_mut() {
                        match &update.entry {
                            QueueEntry::IntradayLearning(entry) => {
                                load_balancer.remove_card(entry.id);
                            }
                            QueueEntry::Main(entry) => {
                                load_balancer.remove_card(entry.id);
                            }
                        }
                    }
                }

                self.save_undo(UndoableQueueChange::CardAnswerUndone(update));

                Ok(())
            }
            UndoableQueueChange::CardAnswerUndone(update) => {
                if let Some(queues) = self.get_or_invalidate_queues(update.queue_build_time)? {
                    queues.pop_entry(update.entry.card_id())?;
                    if let Some(learning) = update.learning_requeue {
                        queues.insert_intraday_learning_card(learning);
                    }
                    queues.restore_cutoff(&update.cutoff_snapshot);
                }
                self.save_undo(UndoableQueueChange::CardAnswered(update));

                Ok(())
            }
            UndoableQueueChange::RwkvCardAnswered(answer) => {
                self.clear_study_queues();
                self.state.rwkv_undone_answer = Some(answer);
                self.save_undo(UndoableQueueChange::RwkvCardAnswerUndone(answer));
                Ok(())
            }
            UndoableQueueChange::RwkvCardAnswerUndone(answer) => {
                self.clear_study_queues();
                self.state.rwkv_undone_answer = None;
                self.save_undo(UndoableQueueChange::RwkvCardAnswered(answer));
                Ok(())
            }
        }
    }

    pub(super) fn save_queue_update_undo(&mut self, change: Box<QueueUpdate>) {
        self.save_undo(UndoableQueueChange::CardAnswered(change))
    }
}

impl CardQueues {
    /// Hayaki: puts rebuilt queues as they were before an undone RWKV answer,
    /// the card on top and learning cards that fell due since kept back. The
    /// card stays out when the rebuilt queues no longer hold it, and nothing
    /// is restored on another day.
    pub(super) fn restore_rwkv_answer(&mut self, answer: RwkvAnswer) {
        if answer.day != self.current_day {
            return;
        }
        if answer.learning_cutoff < self.current_learning_cutoff {
            let ahead = self
                .current_learning_cutoff
                .adding_secs(self.learn_ahead_secs);
            let earlier_ahead = answer.learning_cutoff.adding_secs(self.learn_ahead_secs);
            let not_yet_due = self
                .intraday_learning
                .iter()
                .filter(|e| e.due > earlier_ahead && e.due <= ahead)
                .count();
            self.counts.learning = self.counts.learning.saturating_sub(not_yet_due);
            self.current_learning_cutoff = answer.learning_cutoff;
        }
        let id = answer.card_id;
        if let Some(pos) = self.intraday_learning.iter().position(|e| e.id == id) {
            let entry = self.intraday_learning.remove(pos).unwrap();
            self.intraday_learning.push_front(entry);
        } else if let Some(pos) = self.main.iter().position(|e| e.id == id) {
            let entry = self.main.remove(pos).unwrap();
            self.main.push_front(entry);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::card::CardQueue;
    use crate::card::CardType;
    use crate::deckconfig::LeechAction;
    use crate::prelude::*;

    fn add_note(col: &mut Collection, with_reverse: bool) -> Result<NoteId> {
        let nt = col
            .get_notetype_by_name("Basic (and reversed card)")?
            .unwrap();
        let mut note = nt.new_note();
        note.set_field(0, "one")?;
        if with_reverse {
            note.set_field(1, "two")?;
        }
        col.add_note(&mut note, DeckId(1))?;
        Ok(note.id)
    }

    #[test]
    fn undo() -> Result<()> {
        // add a note
        let mut col = Collection::new();
        let nid = add_note(&mut col, true)?;

        // turn burying and leech suspension on
        let mut conf = col.storage.get_deck_config(DeckConfigId(1))?.unwrap();
        conf.inner.bury_new = true;
        conf.inner.leech_action = LeechAction::Suspend as i32;
        col.storage.update_deck_conf(&conf)?;

        // get the first card
        let queued = col.get_next_card()?.unwrap();
        let cid = queued.card.id;
        let sibling_cid = col.storage.all_card_ids_of_note_in_template_order(nid)?[1];

        let assert_initial_state = |col: &mut Collection| -> Result<()> {
            let first = col.storage.get_card(cid)?.unwrap();
            assert_eq!(first.queue, CardQueue::New);
            let sibling = col.storage.get_card(sibling_cid)?.unwrap();
            assert_eq!(sibling.queue, CardQueue::New);
            Ok(())
        };

        assert_initial_state(&mut col)?;

        // immediately graduate the first card
        col.answer_easy();

        // the sibling will be buried
        let sibling = col.storage.get_card(sibling_cid)?.unwrap();
        assert_eq!(sibling.queue, CardQueue::SchedBuried);

        // make it due now, with 7 lapses. we use the storage layer directly,
        // bypassing undo
        let mut card = col.storage.get_card(cid)?.unwrap();
        assert_eq!(card.ctype, CardType::Review);
        card.lapses = 7;
        card.due = 0;
        col.storage.update_card(&card)?;

        // fail it, which should cause it to be marked as a leech
        col.clear_study_queues();
        col.answer_again();

        let assert_post_review_state = |col: &mut Collection| -> Result<()> {
            let card = col.storage.get_card(cid)?.unwrap();
            assert_eq!(card.interval, 1);
            assert_eq!(card.lapses, 8);

            assert_eq!(
                col.storage.get_all_revlog_entries(TimestampSecs(0))?.len(),
                2
            );

            let note = col.storage.get_note(nid)?.unwrap();
            assert_eq!(note.tags, vec!["leech".to_string()]);
            assert!(!col.storage.all_tags()?.is_empty());

            let deck = col.get_deck(DeckId(1))?.unwrap();
            assert_eq!(deck.common.review_studied, 1);

            assert!(col.get_next_card()?.is_none());

            Ok(())
        };

        let assert_pre_review_state = |col: &mut Collection| -> Result<()> {
            // the card should have its old state, but a new mtime (which we can't
            // easily test without waiting)
            let card = col.storage.get_card(cid)?.unwrap();
            assert_eq!(card.interval, 4);
            assert_eq!(card.lapses, 7);

            // the revlog entry should have been removed
            assert_eq!(
                col.storage.get_all_revlog_entries(TimestampSecs(0))?.len(),
                1
            );

            // the note should no longer be tagged as a leech
            let note = col.storage.get_note(nid)?.unwrap();
            assert!(note.tags.is_empty());
            assert!(col.storage.all_tags()?.is_empty());

            let deck = col.get_deck(DeckId(1))?.unwrap();
            assert_eq!(deck.common.review_studied, 0);
            assert!(col.get_next_card()?.is_some());
            assert_eq!(col.counts(), [0, 0, 1]);

            Ok(())
        };

        // ensure everything is restored on undo/redo
        assert_post_review_state(&mut col)?;
        col.undo()?;
        assert_pre_review_state(&mut col)?;
        col.redo()?;
        assert_post_review_state(&mut col)?;
        col.undo()?;
        assert_pre_review_state(&mut col)?;
        col.undo()?;
        assert_initial_state(&mut col)?;

        Ok(())
    }

    #[test]
    fn undo_counts() -> Result<()> {
        let mut col = Collection::new();
        if col.timing_today()?.near_cutoff() {
            return Ok(());
        }

        assert_eq!(col.counts(), [0, 0, 0]);
        add_note(&mut col, true)?;
        assert_eq!(col.counts(), [2, 0, 0]);
        col.answer_again();
        assert_eq!(col.counts(), [1, 1, 0]);
        col.answer_good();
        assert_eq!(col.counts(), [0, 2, 0]);
        col.answer_again();
        assert_eq!(col.counts(), [0, 2, 0]);
        // first card graduates
        col.answer_good();
        assert_eq!(col.counts(), [0, 1, 0]);
        col.answer_easy();
        assert_eq!(col.counts(), [0, 0, 0]);

        // now work backwards
        col.undo()?;
        assert_eq!(col.counts(), [0, 1, 0]);
        col.undo()?;
        assert_eq!(col.counts(), [0, 2, 0]);
        col.undo()?;
        assert_eq!(col.counts(), [0, 2, 0]);
        col.undo()?;
        assert_eq!(col.counts(), [1, 1, 0]);
        col.undo()?;
        assert_eq!(col.counts(), [2, 0, 0]);
        col.undo()?;
        assert_eq!(col.counts(), [0, 0, 0]);

        // and forwards again
        col.redo()?;
        assert_eq!(col.counts(), [2, 0, 0]);
        col.redo()?;
        assert_eq!(col.counts(), [1, 1, 0]);
        col.redo()?;
        assert_eq!(col.counts(), [0, 2, 0]);
        col.redo()?;
        assert_eq!(col.counts(), [0, 2, 0]);
        col.redo()?;
        assert_eq!(col.counts(), [0, 1, 0]);
        col.redo()?;
        assert_eq!(col.counts(), [0, 0, 0]);

        Ok(())
    }

    #[test]
    fn redo_after_queue_invalidation_bug() -> Result<()> {
        // add a note to the default deck
        let mut col = Collection::new();
        let _nid = add_note(&mut col, true)?;

        // add a deck and select it
        let mut deck = Deck::new_normal();
        deck.name = NativeDeckName::from_human_name("foo");
        col.add_deck(&mut deck)?;
        col.set_current_deck(deck.id)?;

        // select default again, which invalidates current queues
        col.set_current_deck(DeckId(1))?;

        // get the first card and answer it
        col.answer_easy();

        // undo answer
        col.undo()?;

        // undo deck select, which invalidates the queues again
        col.undo()?;

        // redo deck select (another invalidation)
        col.redo()?;

        // when the card answer is redone, it shouldn't fail because
        // the queues are rebuilt after the card state is restored
        col.redo()?;

        Ok(())
    }
}
