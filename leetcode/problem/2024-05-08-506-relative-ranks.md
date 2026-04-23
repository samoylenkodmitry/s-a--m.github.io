---
layout: leetcode-entry
title: "506. Relative Ranks"
permalink: "/leetcode/problem/2024-05-08-506-relative-ranks/"
leetcode_ui: true
entry_slug: "2024-05-08-506-relative-ranks"
---

[506. Relative Ranks](https://leetcode.com/problems/relative-ranks/description/) easy
[blog post](https://leetcode.com/problems/relative-ranks/solutions/5128403/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08052024-506-relative-ranks?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/J2MVn8NVTHo)
![2024-05-08_08-04.webp](/assets/leetcode_daily_images/88aea9ee.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/596

#### Problem TLDR

Convert results array to ranks array #easy #sorting

#### Intuition

Understand what the problem is:
```j
4 3 2 1 -> "4" "Bronze" "Silver" "Gold
```
We need to convert each result with it's position in a sorted order.
There are several ways to do this: use a HashMap, Priority Queue, or just sort twice.

#### Approach

Let's try to write the minimum lines of code version.

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findRelativeRanks(score: IntArray): Array<String> {
        val medals = listOf("Gold", "Silver", "Bronze")
        val inds = score.indices.sortedByDescending { score[it] }
        return inds.indices.sortedBy { inds[it] }.map {
            if (it > 2) "${ it + 1 }" else "${ medals[it] } Medal"
        }.toTypedArray()
    }

```
```rust

    pub fn find_relative_ranks(score: Vec<i32>) -> Vec<String> {
        let mut inds: Vec<_> = (0..score.len()).collect();
        inds.sort_unstable_by_key(|&i| Reverse(score[i]));
        let (mut res, medals) = (inds.clone(), vec!["Gold", "Silver", "Bronze"]);
        res.sort_unstable_by_key(|&r| inds[r]);
        res.iter().map(|&place| if place > 2 { format!("{}", place + 1) }
            else { format!("{} Medal", medals[place]) }).collect()
    }

```

