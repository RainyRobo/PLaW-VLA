import numpy as np
import pandas as pd
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_repack_transform_preserves_is_pad_metadata():
    transform = _transforms.RepackTransform(
        structure={
            "observation/image": "observation.images.image",
        }
    )

    item = {
        "observation.images.image": np.zeros((5, 2, 1, 1), dtype=np.uint8),
        "observation.images.image_is_pad": np.array([True, False, False, False, True], dtype=bool),
    }

    result = transform(item)

    np.testing.assert_array_equal(
        result["observation/image_is_pad"],
        np.array([True, False, False, False, True], dtype=bool),
    )


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_canonical_ee_delta_actions_round_trip():
    quarter_turn_z = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    state = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.25], dtype=np.float32)
    absolute_actions = np.array(
        [
            [2.0, 4.0, 6.0, *quarter_turn_z, 1.0],
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.25],
        ],
        dtype=np.float32,
    )

    delta = _transforms.DeltaActions(ee_pose=True)({"state": state.copy(), "actions": absolute_actions.copy()})["actions"]
    restored = _transforms.AbsoluteActions(ee_pose=True)({"state": state.copy(), "actions": delta.copy()})["actions"]

    np.testing.assert_allclose(delta[0, :3], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(delta[0, 3:7], quarter_turn_z, atol=1e-6)
    np.testing.assert_allclose(delta[0, 7], np.array(0.75, dtype=np.float32))
    np.testing.assert_allclose(delta[1], np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(restored, absolute_actions, atol=1e-6)


def test_canonical_ee_delta_actions_handles_bimanual_prefix_mask():
    state = np.array(
        [
            1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.2,
            4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.8,
            9.0, 9.0,
        ],
        dtype=np.float32,
    )
    absolute_actions = np.array(
        [[
            2.0, 4.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.4,
            5.0, 7.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.1,
            3.0, -2.0,
        ]],
        dtype=np.float32,
    )

    mask = _transforms.make_bool_mask(16, -2)
    delta = _transforms.DeltaActions(mask=mask, ee_pose=True)({"state": state.copy(), "actions": absolute_actions.copy()})["actions"]
    restored = _transforms.AbsoluteActions(mask=mask, ee_pose=True)({"state": state.copy(), "actions": delta.copy()})["actions"]

    np.testing.assert_allclose(delta[0, :3], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(delta[0, 7], np.array(0.2, dtype=np.float32))
    np.testing.assert_allclose(delta[0, 8:11], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(delta[0, 15], np.array(-0.7, dtype=np.float32))
    np.testing.assert_allclose(delta[0, 16:], np.array([3.0, -2.0], dtype=np.float32))
    np.testing.assert_allclose(restored, absolute_actions, atol=1e-6)


def test_canonical_ee_delta_uses_current_state_for_every_chunk_step():
    state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)
    absolute_actions = np.array(
        [
            [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4],
            [0.3, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )

    delta = _transforms.DeltaActions(ee_pose=True)(
        {"state": state.copy(), "actions": absolute_actions.copy()}
    )["actions"]

    np.testing.assert_allclose(delta[:, :3], absolute_actions[:, :3] - state[:3])
    np.testing.assert_allclose(delta[:, 7], absolute_actions[:, 7] - state[7])


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_extract_prompt_from_task_dataframe():
    tasks = pd.DataFrame(
        {"task_index": [0, 1]},
        index=pd.Index(["pick up the cup", "place the cup down"], name="task"),
    )
    transform = _transforms.PromptFromLeRobotTask(tasks)

    data = transform({"task_index": 1})
    assert data["prompt"] == "place the cup down"


def test_split_temporal_frames_splits_full_window():
    frames = np.arange(5 * 2).reshape(5, 2, 1, 1)
    transform = _transforms.SplitTemporalFrames(frame_indices=(-2, -1, 0, 1, 2), image_keys=("image",))

    data = transform({"image": frames})

    np.testing.assert_array_equal(data["image_current"], frames[2])
    np.testing.assert_array_equal(data["image_history"], frames[:3])
    np.testing.assert_array_equal(data["image_future"], frames[3:])


def test_split_temporal_frames_accepts_history_only_inference_input():
    history_only = np.arange(2 * 2).reshape(2, 2, 1, 1)
    transform = _transforms.SplitTemporalFrames(frame_indices=(-2, -1, 0, 1, 2), image_keys=("image",))

    data = transform({"image": history_only})

    np.testing.assert_array_equal(data["image_current"], history_only[-1])
    np.testing.assert_array_equal(data["image_history"], history_only)
    assert data["image_future"].shape[0] == 0


def test_split_temporal_valid_mask_splits_lerobot_padding_metadata():
    history_mask, future_mask = _transforms.split_temporal_valid_mask(
        {
            "image_is_pad": np.array([True, False, False, False, True], dtype=bool),
        },
        "image",
        history_len=3,
        future_len=2,
    )

    np.testing.assert_array_equal(history_mask, np.array([False, True, True], dtype=bool))
    np.testing.assert_array_equal(future_mask, np.array([True, False], dtype=bool))
