---
layout: leetcode-entry
title: "1043. Partition Array for Maximum Sum"
permalink: "/leetcode/problem/2024-02-03-1043-partition-array-for-maximum-sum/"
leetcode_ui: true
entry_slug: "2024-02-03-1043-partition-array-for-maximum-sum"
---

[1043. Partition Array for Maximum Sum](https://leetcode.com/problems/partition-array-for-maximum-sum/description) medium
[blog post](https://leetcode.com/problems/partition-array-for-maximum-sum/solutions/4669799/kotlin-rust/)
[substack](https://dmitriisamoilenko.substack.com/publish/posts/detail/141333259/share-center)
[youtube](https://youtu.be/A4LDXHos0Ho)
![image.png](/assets/leetcode_daily_images/4021e441.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/493

#### Problem TLDR

Max sum of partition array into chunks size of at most k filled with max value in chunk.

#### Intuition

Let's just brute force with Depth-First Search starting from each `i` position: search for the end of chunk `j` and choose the maximum of the sum. max_sum[i] = optimal_chunk + max_sum[chunk_len]. This can be cached by the `i`.

Then rewrite into bottom up DP.

#### Approach

* use size + 1 for dp, to avoid 'if's
* careful with the problem definition: it is not the max count of chunks, it is the chunks lengths up to `k`

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun maxSumAfterPartitioning(arr: IntArray, k: Int): Int {
    val dp = IntArray(arr.size + 1)
    for (i in arr.indices) {
      var max = 0
      for (j in i downTo max(0, i - k + 1)) {
        max = max(max, arr[j])
        dp[i + 1] = max(dp[i + 1], (i - j + 1) * max + dp[j])
      }
    }
    return dp[arr.size]
  }

```
```rust

  pub fn max_sum_after_partitioning(arr: Vec<i32>, k: i32) -> i32 {
    let mut dp = vec![0; arr.len() + 1];
    for i in 0..arr.len() {
      let mut max_v = 0;
      for j in (0..=i).rev().take(k as usize) {
        max_v = max_v.max(arr[j]);
        dp[i + 1] = dp[i + 1].max((i - j + 1) as i32 * max_v + dp[j]);
      }
    }
    dp[arr.len()]
  }

```

