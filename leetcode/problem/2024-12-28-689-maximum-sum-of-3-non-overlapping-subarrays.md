---
layout: leetcode-entry
title: "689. Maximum Sum of 3 Non-Overlapping Subarrays"
permalink: "/leetcode/problem/2024-12-28-689-maximum-sum-of-3-non-overlapping-subarrays/"
leetcode_ui: true
entry_slug: "2024-12-28-689-maximum-sum-of-3-non-overlapping-subarrays"
---

[689. Maximum Sum of 3 Non-Overlapping Subarrays](https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/description/) hard
[blog post](https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/solutions/6197019/kotlin-rust-by-samoylenkodmitry-c0rj/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28122024-689-maximum-sum-of-3-non?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9sp46IOXUK8)
[deep-dive](https://notebooklm.google.com/notebook/b9226ee0-005a-43aa-86ec-5861944d0175/audio)
![1.webp](/assets/leetcode_daily_images/463aff6f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/846

#### Problem TLDR

3 max non-intersecting intervals #hard #dynamic_programming #sliding_window

#### Intuition

Failed to solve.

The naive DFS+memo with searching for best `k` intervals starting with `i` gives TLE.

Now, what working solutions are:
1. Sliding window: slide 3 window `0..k`, `k..2k`, `2k..3k` together. The left window just search for it's max sum. The middle search for `max_left + max_middle`. And the right search for `max_middle + max_right`. Update indices on every update of maximum.
2. Dynamic Programming: `dp[i][c]` is (max_sum, start_ind) for `c` k-subarrays in `0..i`. Then restore parents.

```j

  // 0 1 2 3 4 5 6 7 8 9
  // ----- ~~~~~ -----
  // 0   2 3   5 6   8
  //   ----- ~~~~~ -----
  //   1   3 4   6 7   9  one loop iteration

```

#### Approach

* give up after 1 hour, then look for solutions

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maxSumOfThreeSubarrays(nums: IntArray, k: Int): IntArray {
        var i1 = intArrayOf(0); var i12 = i1 + 0; var i123 = i12 + 0
        var s1 = 0; var s2 = 0; var s3 = 0; var m1 = 0; var m12 = 0; var m123 = 0
        for (i in nums.indices) {
            s1 += (nums.getOrNull(i - 2 * k) ?: 0) - (nums.getOrNull(i - 3 * k) ?: 0)
            s2 += (nums.getOrNull(i - k) ?: 0) - (nums.getOrNull(i - 2 * k) ?: 0)
            s3 += nums[i] - (nums.getOrNull(i - k) ?: 0)
            if (s1 > m1) { m1 = s1; i1[0] = i - 3 * k + 1 }
            if (m1 + s2 > m12) { m12 = m1 + s2; i12 = i1 + (i - 2 * k + 1) }
            if (m12 + s3 > m123) { m123 = m12 + s3; i123 = i12 + (i - k + 1) }
        }
        return i123
    }

```
```rust

    pub fn max_sum_of_three_subarrays(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let k = k as usize;
        let (mut i1, mut i12, mut i123) = (0, (0, k), [0, k, 2 * k]);
        let mut s1 = nums[0..k].iter().sum::<i32>();
        let mut s2 = nums[k..2 * k].iter().sum::<i32>();
        let mut s3 = nums[2 * k..3 * k].iter().sum::<i32>();
        let (mut m1, mut m12, mut m123) = (s1, s1 + s2, s1 + s2 + s3);
        for i in 3 * k..nums.len() {
            s1 += nums[i - 2 * k] - nums[i - 3 * k];
            s2 += nums[i - k] - nums[i - 2 * k];
            s3 += nums[i] - nums[i - k];
            if s1 > m1 { m1 = s1; i1 = i - 3 * k + 1 }
            if m1 + s2 > m12 { m12 = m1 + s2; i12 = (i1, i - 2 * k + 1) }
            if m12 + s3 > m123 { m123 = m12 + s3; i123 = [i12.0, i12.1, i - k + 1] }
        }; i123.iter().map(|&x| x as i32).collect()
    }

```
```c++

    vector<int> maxSumOfThreeSubarrays(vector<int>& nums, int k) {
        int n = nums.size(); vector<int> pref(n + 1, 0);
        for (int i = 1; i <= n; ++i) pref[i] = pref[i - 1] + nums[i - 1];
        vector<vector<vector<int>>>dp(n + 1, vector<vector<int>>(4, vector<int>(2, 0)));
        int mx = 0, pos = -1;
        for (int c = 1; c <= 3; ++c) for (int i = k; i <= n; ++i) {
            dp[i][c][0] = dp[i - 1][c][0];
            dp[i][c][1] = dp[i - 1][c][1];
            int sum = pref[i] - pref[i - k];
            if (dp[i][c][0] < dp[i - k][c - 1][0] + sum)
                dp[i][c][0] = dp[i - k][c - 1][0] + sum, dp[i][c][1] = i;
            if (dp[i][c][0] > mx) mx = dp[i][c][0], pos = dp[i][c][1];
        }
        vector<int> res(3, 0); for (int i = 3; i; --i)
            res[i - 1] = pos - k, pos = dp[pos - k][i - 1][1];
        return res;
    }

```

