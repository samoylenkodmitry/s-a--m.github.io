---
layout: leetcode-entry
title: "3577. Count the Number of Computer Unlocking Permutations"
permalink: "/leetcode/problem/2025-12-10-3577-count-the-number-of-computer-unlocking-permutations/"
leetcode_ui: true
entry_slug: "2025-12-10-3577-count-the-number-of-computer-unlocking-permutations"
---

[3577. Count the Number of Computer Unlocking Permutations](https://leetcode.com/problems/count-the-number-of-computer-unlocking-permutations/) medium
[blog post](https://leetcode.com/problems/count-the-number-of-computer-unlocking-permutations/solutions/7404358/kotlin-rust-by-samoylenkodmitry-g60q/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10122025-3577-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5GZxigPGKtk)

![cd2d2aa9-4cd2-401c-832d-247fbec96810 (1).webp](/assets/leetcode_daily_images/e1b9cfb2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1199

#### Problem TLDR

Permutations to unlock all c[j] right and bigger than c[i] #medium #combinatorics

#### Intuition

```j
    // 123456
    // 1 then take all values bigger as second position
    // 1 p[23456]
    // 12 p[3456]
    // 13 p[2456], the number is n!
    // what if we have duplicates?
    // 1223456
    // we are not allowed to sort
    // 1654223
    // 1 [654223] but every number can be at any place
    //           so it doesnt matter
    // so it is (n-1)!
    // how to  calc factorial of 10^5?
    //
    //
```

* first should be the smallest
* other numbers doesn't matter

#### Approach

* use longs

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 9ms
    fun countPermutations(c: IntArray) = (1..<c.size)
    .fold(1L) { r, t -> if (c[t] > c[0]) (r*t) % 1000000007 else 0 }
```
```rust
// 0ms
    pub fn count_permutations(c: Vec<i32>) -> i32 {
        (1..c.len()).fold(1, |r,t|
        if c[t] > c[0] {(r*t)%1000000007} else { 0 })as _
    }
```

