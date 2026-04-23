---
layout: leetcode-entry
title: "1545. Find Kth Bit in Nth Binary String"
permalink: "/leetcode/problem/2026-03-03-1545-find-kth-bit-in-nth-binary-string/"
leetcode_ui: true
entry_slug: "2026-03-03-1545-find-kth-bit-in-nth-binary-string"
---

[1545. Find Kth Bit in Nth Binary String](https://open.substack.com/pub/dmitriisamoilenko/p/03032026-1545-find-kth-bit-in-nth?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/03032026-1545-find-kth-bit-in-nth?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03032026-1545-find-kth-bit-in-nth?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KxoF88DW9jE)

![cb47841f-071e-4cef-a28c-043dba8bcf89 (1).webp](/assets/leetcode_daily_images/24f2fd37.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1286

#### Problem TLDR

Kth bit in s=s+1+inv(rev(s)) sequence #medium

#### Intuition

```j
    // 2^20 is 1M
    //
    //  011100110110001  15
    //  0111001 1 0110001  7-1-7
    //  a       1 inv(rev(a))
    //  011 1 001          3-1-3
    //  0 1 1              1-1-1
    //
    //  011100110110001  15
    //                   15-1-15 = 31
    //                   31-1-31 = 63
    //                   63-1-63 = 127
    // 1 3 7 15 31 63 127
    // 2^n-1
    //
    // 0
    // 0 1 1
    // 011 1 001
    // 0111001 1 0110001
    //             ^
    //     ^
    //   ^
```

* current middle of K is the highest one bit, 110 - 100
* reverse jump: k = 2*k.takeHighestOneBit()-k
* the glue-positions are all powers of two

O(1) intuition (is just observation of the patterns):
123456789101112
0111001101 1 0
 1 1   1       blue bits are powers of two
0 1 0 1 0  1   odd positions are alterating bits sequence (infinite)
  3  *       * even positions have all the same odd 'core' by dividing by 2 (+flip)
    5    10

* rule 1: odd k is alterating bits 01010101, just return k/2%2
* rule 2: even k goes to the core by dividing /2 until odd met, then it is rule1+flip

#### Approach

* start with drawing the patterns
* look for the unique properties, how many you can spot, useful or not
* use small examples to test theory, k = 0, 1, 2, 3
* use one big example to prove it still works k = 11
* spend another 2 hours asking ai to give as many other ideas as possible

#### Complexity

- Time complexity:
$$O(log(k))$$ or O(1)

- Space complexity:
$$O(log(k))$$ or O(1)

#### Code

```kotlin
// 0ms
    fun findKthBit(n: Int, k: Int): Char =
        if (k < 2) '0' else if (k and (k-1)==0) '1' else
        '1'+ ('0'-findKthBit(n, 2*k.takeHighestOneBit()-k))
```
```rust
// 0ms
    pub fn find_kth_bit(n: i32, mut k: i32) -> char {
        (b'0' + if k % 2 > 0 { k/2 % 2 } else { 1-(k >> k.trailing_zeros())/2%2 }as u8) as _
    }
```

