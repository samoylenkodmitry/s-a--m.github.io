---
layout: leetcode-entry
title: "3514. Number of Unique XOR Triplets II"
permalink: "/leetcode/problem/2026-07-24-3514-number-of-unique-xor-triplets-ii/"
leetcode_ui: true
entry_slug: "2026-07-24-3514-number-of-unique-xor-triplets-ii"
---

[3514. Number of Unique XOR Triplets II](https://leetcode.com/problems/number-of-unique-xor-triplets-ii/solutions/8416716/kotlin-rust-by-samoylenkodmitry-09gn/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24072026-3514-number-of-unique-xor?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/g8eSf8yuXjQ)

https://dmitrysamoylenko.com/leetcode/

![24.07.2026.webp](/assets/leetcode_daily_images/24.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1430

#### Problem TLDR

Uniq xor of tripplets in array

#### Intuition

The number of uniq xors is very compressable, we can assume that converting n^2 pairwise to set gives a small colleciton

#### Approach

* to speed up more use boolean array/bitset

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
    fun uniqueXorTriplets(n: IntArray) = buildSet {
        for (x in buildSet { for (x in n) for (y in n) add(x xor y) })
            for (y in n) add(x xor y) }.size
```
```rust
    pub fn unique_xor_triplets(mut n: Vec<i32>) -> i32 {
        n.sort_unstable(); n.dedup();
        let mut s = [false; 1 << 12]; let mut t = s.clone();
        for i in 0..n.len() { for j in i..n.len() { s[(n[i] ^ n[j]) as usize] = true }}
        for x in 0..s.len() { if s[x] { for &v in &n { t[x ^ v as usize] = true }}}
        t.iter().filter(|&&b| b).count() as _
    }
```

