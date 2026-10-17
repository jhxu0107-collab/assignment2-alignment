
import torch

advantages= []
def compute_group_normalized_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
        group_size,
        advantage_eps,
        normalize_by_std
):
        # take raw reward out
        raw_rewards = []
        # for each rollout response, match with gt
        for resp, gt in zip(rollout_responses, repeated_ground_truths):
            reward = reward_fn(resp, gt)["reward"]  # score this response
            raw_rewards.append(reward) # add to the list
        raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)

        # compute within the group
        grouped = raw_rewards.view(-1, group_size)

        # compute A with group normalization
        # shape of reward is (group_number, response number)
        group_mean = grouped.mean(dim=1, keepdim=True)  # (n_groups, 1)
        if normalize_by_std:
            group_std = grouped.std(dim=1, keepdim=True)
            advantages = (grouped - group_mean) / (group_std + advantage_eps)
        else:
            advantages = grouped - group_mean  # Dr. GRPO variant, Eq. 31

        #  back to one value per response
        advantages = advantages.view(-1)  # (rollout_batch_size,)

        metadata = {
            "reward_mean": raw_rewards.mean().item(),
            "reward_std": raw_rewards.std().item(),
            "reward_max": raw_rewards.max().item(),
            "reward_min": raw_rewards.min().item(),
        }
        return advantages, raw_rewards, metadata


