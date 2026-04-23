---
layout: leetcode-entry
title: "3075. Maximize Happiness of Selected Children"
permalink: "/leetcode/problem/2024-05-09-3075-maximize-happiness-of-selected-children/"
leetcode_ui: true
entry_slug: "2024-05-09-3075-maximize-happiness-of-selected-children"
---

[3075. Maximize Happiness of Selected Children](https://leetcode.com/problems/maximize-happiness-of-selected-children/description/) medium
[blog post](https://leetcode.com/problems/maximize-happiness-of-selected-children/solutions/5134240/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09052024-3075-maximize-happiness?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/c5Vv4uRUrYU)
![2024-05-09_11-24.webp](/assets/leetcode_daily_images/47902ed4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/597

#### Problem TLDR

Sum of `k` maximums decreasing each step #medium #sorting #heap #quickselect

#### Intuition

By the problem definition we may assume that optimal solution is to take the largest values first, as smaller values will not decrease the result after reaching zero.

There are several ways to take `k` largest values: sort the entire array, use Heap (PriorityQueue) or use QuickSelect and sort partially.

#### Approach

Let's use PriorityQueue in Kotlin (`min heap`) and QuickSelect in Rust (`select_nth_unstable`).
* when using heap we can take at most `k` values into it to save space and time
* Rust's `select_nth_unstable` result tuple is not very easy to use (do you know a better way?)

#### Complexity

- Time complexity:
$$O(n + klog(k))$$ for the Heap and for the QuickSelect

- Space complexity:
$$O(n)$$ for the Heap, $$O(1)$$ for the QuickSelect

#### Code

```kotlin

    fun maximumHappinessSum(happiness: IntArray, k: Int): Long {
        val pq = PriorityQueue<Int>()
        for (h in happiness) { pq += h; if (pq.size > k) pq.poll() }
        return (0..<k).sumOf { max(0, pq.poll() + it - k + 1).toLong() }
    }

```
```rust

    pub fn maximum_happiness_sum(mut happiness: Vec<i32>, k: i32) -> i64 {
        let count = 0.max(happiness.len() as i32 - k - 1) as usize;
        let gt = if count > 0 { happiness.select_nth_unstable(count).2 }
                 else { &mut happiness[..] };
        gt.sort_unstable_by(|a, b| b.cmp(a));
        (0..k).map(|i| 0.max(gt[i as usize] - i) as i64).sum()
    }

```

