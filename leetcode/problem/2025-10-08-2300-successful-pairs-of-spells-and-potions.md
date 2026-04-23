---
layout: leetcode-entry
title: "2300. Successful Pairs of Spells and Potions"
permalink: "/leetcode/problem/2025-10-08-2300-successful-pairs-of-spells-and-potions/"
leetcode_ui: true
entry_slug: "2025-10-08-2300-successful-pairs-of-spells-and-potions"
---

[2300. Successful Pairs of Spells and Potions](https://leetcode.com/problems/successful-pairs-of-spells-and-potions/description) medium
[blog post](https://leetcode.com/problems/successful-pairs-of-spells-and-potions/solutions/7258583/kotlin-rust-by-samoylenkodmitry-68jz/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05102025-2300-successful-pairs-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/thUncq1nBzQ)

![52c5b1de-7775-444b-971c-02899fbf9a0f (2).webp](/assets/leetcode_daily_images/35379f55.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1136

#### Problem TLDR

Number of potions * spells[i] bigger than `success` #midium #bs #sort

#### Intuition

Sort potions. Binary search value `spells[i] * potions[j] < success`.

#### Approach

* or search for `success + spells - 1 / spells` withoud converting to double

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 105ms
    fun successfulPairs(sp: IntArray, p: IntArray, s: Long) = run { p.sort()
        sp.map { v -> 1 + p.size + p.asList().binarySearch{ if (1L*it*v < s) -1 else 1 }}}

```
```rust

// 17ms
    pub fn successful_pairs(sp: Vec<i32>, mut p: Vec<i32>, s: i64) -> Vec<i32> {
        p.sort_unstable(); let l = p.len() as i32;
        sp.iter().map(|&v| l - p.partition_point(|&p| v as i64 * (p as i64) < s) as i32).collect()
    }

```

