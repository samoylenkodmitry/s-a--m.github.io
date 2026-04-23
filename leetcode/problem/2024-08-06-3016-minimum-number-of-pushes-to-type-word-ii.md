---
layout: leetcode-entry
title: "3016. Minimum Number of Pushes to Type Word II"
permalink: "/leetcode/problem/2024-08-06-3016-minimum-number-of-pushes-to-type-word-ii/"
leetcode_ui: true
entry_slug: "2024-08-06-3016-minimum-number-of-pushes-to-type-word-ii"
---

[3016. Minimum Number of Pushes to Type Word II](https://leetcode.com/problems/minimum-number-of-pushes-to-type-word-ii/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-pushes-to-type-word-ii/solutions/5594921/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06082024-3016-minimum-number-of-pushes?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5U4Q0dnSlrA)
![1.webp](/assets/leetcode_daily_images/6195bdbd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/694

#### Problem TLDR

Minimum keystrokes after assigning letter to keys #medium

#### Intuition

By intuition we should assign the more frequent letters first.

#### Approach

We can use some languages' API, or math `(i / 8 + 1)`.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minimumPushes(word: String) = word
        .groupingBy { it }.eachCount()
        .values.sortedDescending()
        .chunked(8).withIndex()
        .sumOf { (i, s) -> (i + 1) * s.sum() }

```
```rust

    pub fn minimum_pushes(word: String) -> i32 {
        let mut freq = vec![0; 26];
        for b in word.bytes() { freq[b as usize - 97] += 1 }
        freq.sort_unstable();
        (0..26).map(|i| (i as i32 / 8 + 1) * freq[25 - i]).sum()
    }

```

