---
layout: leetcode-entry
title: "2285. Maximum Total Importance of Roads"
permalink: "/leetcode/problem/2024-06-28-2285-maximum-total-importance-of-roads/"
leetcode_ui: true
entry_slug: "2024-06-28-2285-maximum-total-importance-of-roads"
---

[2285. Maximum Total Importance of Roads](https://leetcode.com/problems/maximum-total-importance-of-roads/description/) medium
[blog post](https://leetcode.com/problems/maximum-total-importance-of-roads/solutions/5380529/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28062024-2285-maximum-total-importance?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/bsOXQ3vwlMg)
![2024-06-28_06-37.webp](/assets/leetcode_daily_images/d650482d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/653

#### Problem TLDR

Sort graph by siblings and compute sum(i*s) #medium

#### Intuition

Notice that the more siblings the bigger rank should be to produce the optimal result.

#### Approach

We can sort the count array or use bucket sort of size n to reduce time complexity to O(n).

#### Complexity

- Time complexity:
$$nlog(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maximumImportance(n: Int, roads: Array<IntArray>): Long {
        val counts = IntArray(n); var i = 1
        for ((a, b) in roads) { counts[a]++; counts[b]++ }
        return counts.sorted().sumOf { it * (i++).toLong() }
    }

```
```rust

    pub fn maximum_importance(n: i32, roads: Vec<Vec<i32>>) -> i64 {
        let mut counts = vec![0; n as usize];
        for r in roads { counts[r[0] as usize] += 1; counts[r[1] as usize] += 1}
        counts.sort_unstable();
        (0..n as usize).map(|i| counts[i] * (i + 1) as i64).sum()
    }

```

