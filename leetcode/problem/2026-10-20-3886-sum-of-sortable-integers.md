---
layout: leetcode-entry
title: "3886. Sum of Sortable Integers"
permalink: "/leetcode/problem/2026-10-20-3886-sum-of-sortable-integers/"
leetcode_ui: true
entry_slug: "2026-10-20-3886-sum-of-sortable-integers"
---

[3886. Sum of Sortable Integers](https://leetcode.com/problems/sum-of-sortable-integers/solutions/8581353/kotlin-rust-by-samoylenkodmitry-adlq/) hard
[substack](https://dmitriisamoilenko.substack.com/p/20102026-3886-sum-of-sortable-integers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NvQ_8-uHuYo)

https://dmitrysamoylenko.com/leetcode/

![20.10.2026.webp](/assets/leetcode_daily_images/20.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1517

#### Problem TLDR

Sum of chunks lengths that could split then cycle to sort array

#### Intuition

Brute-force: for every divisor, check if chunk can be sorted by cycling and that previous chunk is less than current

#### Approach

* join 2 junks and count longest sorted sequence
* compare previous chunk max with current chunk min

#### Complexity

- Time complexity:
$$O(nsqrt(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun sortableIntegers(n: IntArray): Int =
    (1..n.size).filter { n.size % it == 0 }.sumOf { k ->
        val c = n.toList().chunked(k)
        if (c.all { b -> b.indices.count { b[it] > b[(it + 1) % k] } <= 1 } &&
            c.zipWithNext().all { (a, b) -> a.max() <= b.min() }) k else 0
    }
```
```rust
    pub fn sortable_integers(n: Vec<i32>) -> i32 {
        (1..=n.len()).filter(|&k| n.len() % k == 0
            && n.chunks(k).all(|c| c.iter().cycle().take(k + 1).tuple_windows().filter(|(a, b)| a > b).count() <= 1)
            && n.chunks(k).tuple_windows().all(|(a, b)| a.iter().max() <= b.iter().min())
        ).sum::<usize>() as _
    }
```

