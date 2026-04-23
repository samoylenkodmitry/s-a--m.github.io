---
layout: leetcode-entry
title: "1395. Count Number of Teams"
permalink: "/leetcode/problem/2024-07-29-1395-count-number-of-teams/"
leetcode_ui: true
entry_slug: "2024-07-29-1395-count-number-of-teams"
---

[1395. Count Number of Teams](https://leetcode.com/problems/count-number-of-teams/description/) medium
[blog post](https://leetcode.com/problems/count-number-of-teams/solutions/5551372/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29072024-1395-count-number-of-teams?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/CCdEKYJvQAc)
![2024-07-29_07-57_1.webp](/assets/leetcode_daily_images/92b32b10.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/685

#### Problem TLDR

Count increasing or decreasing `(i, j, k)` #medium

#### Intuition

The brute-force n^3 solution is accepted.
Now, let's think about the optimization. One way is to precompute some `less[i]` and `bigger[i]` arrays in O(n^2).
Another way is to just multiply count to the left and count to the right.

#### Approach

* just count the lesser values, the bigger will be all the others
* on the right side, just do the additions of the left counts

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun numTeams(rating: IntArray) =
        rating.withIndex().sumOf { (i, r) ->
            var s = (0..<i).count { rating[it] < r }
            (i + 1..<rating.size).sumOf { if (rating[it] < r) i - s else s }
        }

```
```rust

    pub fn num_teams(rating: Vec<i32>) -> i32 {
        (0..rating.len()).map(|i| {
            let s = (0..i).filter(|&j| rating[j] < rating[i]).count();
            (i + 1..rating.len()).map(|j|
                if rating[j] < rating[i] { i - s } else { s } as i32
            ).sum::<i32>()
        }).sum()
    }

```

