---
layout: leetcode-entry
title: "3304. Find the K-th Character in String Game I"
permalink: "/leetcode/problem/2025-07-03-3304-find-the-k-th-character-in-string-game-i/"
leetcode_ui: true
entry_slug: "2025-07-03-3304-find-the-k-th-character-in-string-game-i"
---

[3304. Find the K-th Character in String Game I](https://leetcode.com/problems/find-the-k-th-character-in-string-game-i/description/) easy
[blog post](https://leetcode.com/problems/find-the-k-th-character-in-string-game-i/solutions/6914362/kotlin-rust-by-samoylenkodmitry-os9r/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/3072025-3304-find-the-k-th-character?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/b6L5AosRrHY)
![1.webp](/assets/leetcode_daily_images/fcb37309.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1038

#### Problem TLDR

`k`th char after appending rotated self #easy

#### Intuition

Simulation is accepted.

Some patterns:
```j
    // 01 23 4567 8910     16               32
    // 01 12 1223 13
    // ab bc bccd bccd.... bccd............ bccd........
    //                 cdde
    //                 12
```
* the `b` is always at pos power of two
* each time we double into the `left part` and `right part`
* left is untouched, right is shifted once against previous
* left is at POS%2 == 0, right is at POS%2 > 0

#### Approach

* do shift at each set bit
* 'a' will never overflow 'z', the first 'z' is at `(1 << 25) - 1` position

#### Complexity

- Time complexity:
$$O(k^2)$$

- Space complexity:
$$O(k)$$

#### Code

```kotlin

// 13ms
    fun kthCharacter(k: Int): Char {
        var s = listOf('a')
        while (s.lastIndex < k - 1) s += s.map { it + 1 }
        return s[k - 1]
    }

```
```kotlin

// 1ms
    fun kthCharacter(k: Int): Char {
        var k = k - 1; var c = 'a'
        while (k > 0) { c += k % 2; k /= 2 }
        return c
    }

```
```kotlin

// 0ms
    fun kthCharacter(k: Int): Char {
        fun x(k: Int): Char = if (k == 0) 'a' else x(k / 2) + k % 2
        return x(k - 1)
    }

```
```kotlin

// 0ms
    fun kthCharacter(k: Int, l: Int = 1): Char =
        if (k == l) 'a' else kthCharacter((k - l) / 2, 0) + (k - l) % 2

```
```kotlin

// 0ms
    fun kthCharacter(k: Int) = 'a' + (k - 1).countOneBits()

```
```rust

// 0ms
    pub fn kth_character(k: i32) -> char {
        "abcdefghi".as_bytes()[(k - 1).count_ones() as usize] as char
    }

```
```c++

// 0ms
    char kthCharacter(int k) {
        return 'a' + __builtin_popcount(k - 1);
    }

```

