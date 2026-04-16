"""Canned benchmark scenarios for the T3.2 prototypes.

The canonical scenario is a 45-minute simulated session that:
1. introduces 8 named entities across the first 5 minutes
2. references those entities in turns 9-45 (minutes 5-35)
3. receives a goAway at ~minute 9 (forcing transparent resume)
4. receives an unexpected close at ~minute 18 (forcing cold reconnect)
5. continues through minute 45 and ends cleanly

Use with harness.mock_client_factory(SCENARIO_45MIN_SCRIPTS, ENTITY_RESPONSE_FN).
"""

from __future__ import annotations

from harness import SessionController

# ----- ground-truth entities the scenario introduces -----

ENTITY_GROUND_TRUTH: list[dict] = [
    {"surface": "田中先生", "en": "Dr. Tanaka", "type": "person"},
    {"surface": "山田さん", "en": "Mr. Yamada", "type": "person"},
    {"surface": "東京大学", "en": "University of Tokyo", "type": "org"},
    {"surface": "京都", "en": "Kyoto", "type": "place"},
    {"surface": "渋谷", "en": "Shibuya", "type": "place"},
    {"surface": "ソニー", "en": "Sony", "type": "org"},
    {"surface": "鈴木部長", "en": "Director Suzuki", "type": "person"},
    {"surface": "新宿駅", "en": "Shinjuku Station", "type": "place"},
]


def entity_response_fn(prompt: str, call_index: int) -> list[dict]:
    """Return entities found in the prompt text, based on ENTITY_GROUND_TRUTH."""
    found: list[dict] = []
    for ent in ENTITY_GROUND_TRUTH:
        if ent["surface"] in prompt:
            # Skip if prompt's "Known list" section indicates this entity is known.
            known_idx = prompt.find("Known list (exclude these):")
            text_idx = prompt.find("Text:\n")
            if known_idx >= 0 and text_idx > known_idx:
                known_block = prompt[known_idx:text_idx]
                if ent["surface"] in known_block:
                    continue
            found.append(dict(ent))
    return found


# ----- the canonical 45-min script split across 3 sessions -----


async def _emit_intro_turns(ctrl: SessionController):
    # Session 1: minutes 0-9. Introduce entities and have some conversation.
    clock = ctrl._clock
    # issue handle early so transparent resume is possible
    await clock.advance(3.0)
    ctrl.issue_handle()
    for sim_minute, (ja, en) in enumerate(
        [
            ("おはようございます、田中先生。", "Good morning, Dr. Tanaka."),
            ("今日は山田さんも来ていますね。", "Mr. Yamada is also here today."),
            ("東京大学で会議があります。", "There's a meeting at the University of Tokyo."),
            ("来週、京都に行く予定です。", "I'm planning to go to Kyoto next week."),
            ("渋谷で待ち合わせしましょう。", "Let's meet up in Shibuya."),
            ("ソニーの新製品を見ました。", "I saw Sony's new product."),
            ("鈴木部長もそう言っていました。", "Director Suzuki said so too."),
            ("新宿駅で乗り換えます。", "I'll transfer at Shinjuku Station."),
        ]
    ):
        ctrl.emit_block(ja, en)
        await clock.advance(60.0)  # one block per simulated minute
        if sim_minute == 3:
            # refresh handle partway through
            ctrl.issue_handle()
    # At simulated minute 9, send goAway.
    ctrl.go_away(time_left="60s")
    # give receiver a moment to notice, then close
    await clock.advance(2.0)
    ctrl.force_close()


async def _emit_followup_turns(ctrl: SessionController):
    # Session 2: minutes 9-18. This is the transparent-resumed session (handle carried).
    # Simulates the model remembering names from prior session via resumption.
    clock = ctrl._clock
    await clock.advance(2.0)
    ctrl.issue_handle()
    for ja, en in [
        ("田中先生、さっきの件ですが、", "Dr. Tanaka, about what we discussed earlier,"),
        ("山田さんの報告を確認しました。", "I've reviewed Mr. Yamada's report."),
        ("京都の会議はいつですか。", "When is the meeting in Kyoto?"),
        ("鈴木部長にも伝えておきます。", "I'll let Director Suzuki know as well."),
        ("渋谷のカフェで話しましょうか。", "Shall we talk at the cafe in Shibuya?"),
        ("東京大学の研究室に行きます。", "I'll go to the University of Tokyo lab."),
        ("ソニーの担当者と電話しました。", "I called Sony's representative."),
        ("新宿駅の東口で待ちます。", "I'll wait at Shinjuku Station's east exit."),
    ]:
        ctrl.emit_block(ja, en)
        await clock.advance(60.0)
    # At simulated minute 18, unexpected close (no goAway).
    ctrl.force_close()


async def _emit_late_turns(ctrl: SessionController):
    # Session 3: minutes 18-45. Cold reconnect — no handle carried (depending on
    # which prototype, may or may not re-establish context).
    clock = ctrl._clock
    await clock.advance(3.0)
    ctrl.issue_handle()
    for ja, en in [
        ("田中先生の意見を聞きたいです。", "I'd like to hear Dr. Tanaka's opinion."),
        ("山田さん、会議の時間は？", "Mr. Yamada, what time is the meeting?"),
        ("京都では何を見ましたか。", "What did you see in Kyoto?"),
        ("鈴木部長は今どこにいますか。", "Where is Director Suzuki right now?"),
        ("東京大学の教授と話しました。", "I spoke with a professor at the University of Tokyo."),
        ("ソニーの株価が上がっています。", "Sony's stock price is rising."),
        ("渋谷のスクランブル交差点は混雑しています。", "The Shibuya scramble crossing is crowded."),
        ("新宿駅から歩いて15分です。", "It's a 15-minute walk from Shinjuku Station."),
        ("田中先生、ありがとうございました。", "Thank you, Dr. Tanaka."),
        ("山田さん、また連絡します。", "Mr. Yamada, I'll contact you again."),
        ("明日の予定を確認しましょう。", "Let's confirm tomorrow's schedule."),
        ("会議の資料を送ります。", "I'll send the meeting materials."),
        ("それでは、失礼します。", "Well then, excuse me."),
    ]:
        ctrl.emit_block(ja, en)
        await clock.advance(60.0)
    # Clean end after minute 45.
    ctrl.force_close()


SCENARIO_45MIN_SCRIPTS = [
    _emit_intro_turns,
    _emit_followup_turns,
    _emit_late_turns,
]


# ----- cold-start scenario: B's seed path only fires when handle is None -----


async def _cold_start_with_prior_history(ctrl: SessionController):
    """Single-session script. Bench pre-populates recent_blocks before calling."""
    clock = ctrl._clock
    await clock.advance(2.0)
    ctrl.issue_handle()
    for ja, en in [
        ("おはようございます、田中先生。", "Good morning, Dr. Tanaka."),
        ("さっきの山田さんの件について。", "About Mr. Yamada's matter from earlier."),
        ("京都に行く準備はできましたか。", "Are you ready to go to Kyoto?"),
    ]:
        ctrl.emit_block(ja, en)
        await clock.advance(30.0)
    ctrl.force_close()


SCENARIO_COLD_START_SCRIPTS = [_cold_start_with_prior_history]


# Pre-populated history for the cold-start scenario. Mimics state restored
# from disk after a process restart. Three entities appear here; B should
# seed them via send_client_content on first connect.
PRIOR_HISTORY_BLOCKS: list[dict] = [
    {"ja": "JA: 田中先生、こんにちは。", "en": "EN: Hello, Dr. Tanaka."},
    {"ja": "JA: 山田さんも来ていますね。", "en": "EN: Mr. Yamada is here too."},
    {"ja": "JA: 東京大学の会議に出席しました。", "en": "EN: I attended the meeting at the University of Tokyo."},
    {"ja": "JA: 来週、京都に行く予定です。", "en": "EN: I plan to go to Kyoto next week."},
]


# ----- helper metrics -----


def count_expected_blocks() -> int:
    return 8 + 8 + 13  # sessions 1, 2, 3


def count_expected_blocks_cold_start() -> int:
    return 3
