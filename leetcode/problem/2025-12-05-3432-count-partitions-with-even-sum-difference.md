---
layout: leetcode-entry
title: "3432. Count Partitions with Even Sum Difference"
permalink: "/leetcode/problem/2025-12-05-3432-count-partitions-with-even-sum-difference/"
leetcode_ui: true
entry_slug: "2025-12-05-3432-count-partitions-with-even-sum-difference"
---

[3432. Count Partitions with Even Sum Difference](https://leetcode.com/problems/count-partitions-with-even-sum-difference/description/) easy
[blog post](https://leetcode.com/problems/count-partitions-with-even-sum-difference/solutions/7393372/kotlin-rust-by-samoylenkodmitry-mlko/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05112025-3432-count-partitions-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7KtVrBXb4DA)

![9d93b705-13db-48c9-9709-31ab503779dd (1).webp](/assets/leetcode_daily_images/3a644f2e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1194

#### Problem TLDR

Count Left sum - Right sum % 2 #easy

#### Intuition

Just brute-force.

#### Approach

* a more interesting solution: A-B %2 == 0 only if both odd or both even.
* step i+0, [.......even_sum][..........even_sum]
* step i+1, [.......even_sum,odd][..........even_sum-odd]
* step i+1, [.......even_sum,even][..........even_sum-even]
* basically, the even-ness will be the same for every partition
* same proof for odd-ness

#### Complexity

- Time complexity:
$$O(n^2)$$, or O(n) for clever solution

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 11ms
    fun countPartitions(n: IntArray) =
        (n.size-1)* (1-n.sum()%2)
```
```rust
// 0ms
    pub fn count_partitions(n: Vec<i32>) -> i32 {
       (n.len()as i32-1)*(1-n.iter().sum::<i32>()%2)
    }
```

