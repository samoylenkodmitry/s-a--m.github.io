---
layout: leetcode-entry
title: "3307. Find the K-th Character in String Game II"
permalink: "/leetcode/problem/2025-07-04-3307-find-the-k-th-character-in-string-game-ii/"
leetcode_ui: true
entry_slug: "2025-07-04-3307-find-the-k-th-character-in-string-game-ii"
---

[3307. Find the K-th Character in String Game II](https://leetcode.com/problems/find-the-k-th-character-in-string-game-ii/description) hard
[blog post](https://leetcode.com/problems/find-the-k-th-character-in-string-game-ii/solutions/6918431/kotlin-rust-by-samoylenkodmitry-tye0/)
[substack](https://dmitriisamoilenko.substack.com/p/4072025-3307-find-the-k-th-character?r=2bam17)
[youtube](https://youtu.be/V22PF_PM5Gw)
![1.webp](/assets/leetcode_daily_images/58f82c56.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1039

#### Problem TLDR

kth char in double+shift by ops[i] string #hard #bit_manipulation

#### Intuition

Took me too long to spot the reversive order of operations.

```j
    // a
    // ab
    // 1234
    // abab

    // 12345678910
    //          *
    //     *
    //  *
    // *

    // 0123456789
    // abbcbccdbc
    // 0112122312
    //   . .    *+1 op = 1  single conversion
    //   . *+0 op=0
    //   *+0 op=1
    //  *+1 op=0
    // *0

    // 0123456789
    // abbcbccdbc    k=3   op=[1,0]
    // 0112122312
    //   *+0 op=0
    //  *+1 op=1
    // *0

    // a - ab op=1
    // ab - abab op=0 notice the reversive order of `op`
    // 0123
    // abab
    //   *
```

* each time string doubles
* the `left` part is always skips shift
* the `right` part do shift if `operations[op] == 1`
* given position `x` can be from the left if `x % 2 == 0` or from the right if `x % 2 == 1`
* as we go from child to parent, operations[] are inversed

#### Approach

* this time overflow of `z` is actually possible, don't forget %26

#### Complexity

- Time complexity:
$$O(o)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 1ms
    fun kthCharacter(k: Long, operations: IntArray): Char {
        var x = 0L; var p = 1L
        for (o in operations) { x += p * o; p *= 2 }
        return 'a' + (x and (k - 1)).countOneBits() % 26
    }

```
```kotlin

// 1ms
    fun kthCharacter(k: Long, operations: IntArray, i: Int = 0, l: Long = 1L): Char =
        if (k == l) 'a' else 'a' + (kthCharacter((k - l) / 2, operations, i + 1, 0) + ((k - l) % 2).toInt() * operations[i] - 'a') % 26

```
```rust

// 0ms
    pub fn kth_character(k: i64, operations: Vec<i32>) -> char {
       let (mut x, mut p) = (0, 1); for o in operations { x += p * o as i64; p *= 2 }
       "abcdefghijklmnopqrstuvwxyz".as_bytes()[((x & (k - 1)).count_ones() % 26) as usize] as char
    }

```
```c++

// 0ms
    char kthCharacter(long long k, vector<int>& o) {
        --k; char c = 'a';
        for (int o: o) c = 'a' + (c + ((k & 1) & o) - 'a') % 26, k /= 2;
        return c;
    }

```

