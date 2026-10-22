---
layout: leetcode-entry
title: "3942. Minimum Operations to Sort a Permutation"
permalink: "/leetcode/problem/2026-10-22-3942-minimum-operations-to-sort-a-permutation/"
leetcode_ui: true
entry_slug: "2026-10-22-3942-minimum-operations-to-sort-a-permutation"
---

[3942. Minimum Operations to Sort a Permutation](https://leetcode.com/problems/minimum-operations-to-sort-a-permutation/solutions/8583441/kotlin-rust-by-samoylenkodmitry-8334/) medium
[substack](https://dmitriisamoilenko.substack.com/p/22102026-3942-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cgvtv-pbEBc)

https://dmitrysamoylenko.com/leetcode/

![22.10.2026.webp](/assets/leetcode_daily_images/22.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1519

#### Problem TLDR

Ops to sort by shifting left and reversing

#### Intuition

Brain teaser of corner cases.
Those operations never shuffle.
```j
    // a b c d
    //   b c d a
    //   a d c b
    //     d c b a
    //       c b a d
    //         b a d c
    //         c d a b
    // number of shifts plus possible inversion
    // find sorted order in cycle
    // numbers are never shuffle
```
Count how many are decreasing/increasing consequently.
Find a point of split.
Check reversed array subproblem.

#### Approach

* numbers are 0..<n, means the array[0] is the shift count

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun minOperations(a: IntArray) = run {
        val i = (1..<a.size).all { a[it] == (a[it - 1] + 1) % a.size }
        val d = (1..<a.size).all { a[it - 1] == (a[it] + 1) % a.size }
        when {
            i && a[0] == 0 -> 0
            i -> min(a.size - a[0], a[0] + 2)
            d -> min(a.last(), a.size - a.last()) + 1
            else -> -1
        }
    }
```
```rust
    pub fn min_operations(a: Vec<i32>) -> i32 {
        let (n, f, l) = (a.len() as i32, a[0], *a.last().unwrap());
        let (i, d) = a.windows(2).fold((0, 0), |(i, d), w| (
            i + (w[1] == (w[0] + 1) % n) as i32,
            d + (w[0] == (w[1] + 1) % n) as i32,
        ));
        if i == n - 1 { if f == 0 { 0 } else { (n - f).min(f + 2) } }
        else if d == n - 1 { l.min(n - l) + 1 }
        else { -1 }
    }
```

