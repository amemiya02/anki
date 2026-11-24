// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html
use std::collections::HashMap;

use crate::prelude::*;

#[derive(Debug)]
pub(crate) struct DueCounts {
    pub new: u32,
    pub review: u32,
    /// interday+intraday
    pub learning: u32,

    pub intraday_learning: u32,
    pub interday_learning: u32,
    pub total_cards: u32,
}

impl Deck {
    /// Return the studied counts if studied today.
    /// May be negative if user has extended limits.
    pub(crate) fn new_rev_counts(&self, today: u32) -> (i32, i32) {
        if self.common.last_day_studied == today {
            (self.common.new_studied, self.common.review_studied)
        } else {
            (0, 0)
        }
    }
}

impl Collection {
    /// Get due counts for decks at the given timestamp.
    pub(crate) fn due_counts(
        &mut self,
        days_elapsed: u32,
        learn_cutoff: u32,
    ) -> Result<HashMap<DeckId, DueCounts>> {
        let mut counts = self.storage.due_counts(days_elapsed, learn_cutoff)?;
        // Hayaki: with RWKV scheduling on and ready, a normal deck's review
        // count is the number of its cards the RWKV rule queues. Filtered
        // decks keep upstream's count.
        if let Some(rwkv) = self.rwkv_review_counts_by_deck()? {
            let decks = self.storage.get_decks_map()?;
            for (deck_id, deck) in &decks {
                if deck.is_filtered() {
                    continue;
                }
                let review = rwkv.get(deck_id).copied().unwrap_or(0);
                match counts.get_mut(deck_id) {
                    Some(entry) => entry.review = review,
                    None if review > 0 => {
                        counts.insert(
                            *deck_id,
                            DueCounts {
                                new: 0,
                                review,
                                learning: 0,
                                intraday_learning: 0,
                                interday_learning: 0,
                                total_cards: 0,
                            },
                        );
                    }
                    None => {}
                }
            }
        }
        Ok(counts)
    }

    pub(crate) fn counts_for_deck_today(
        &mut self,
        did: DeckId,
    ) -> Result<anki_proto::scheduler::CountsForDeckTodayResponse> {
        let today = self.current_due_day(0)?;
        let mut deck = self.storage.get_deck(did)?.or_not_found(did)?;
        deck.reset_stats_if_day_changed(today);
        Ok(anki_proto::scheduler::CountsForDeckTodayResponse {
            new: deck.common.new_studied,
            review: deck.common.review_studied,
        })
    }
}
