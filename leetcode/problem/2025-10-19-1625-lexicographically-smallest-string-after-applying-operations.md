---
layout: leetcode-entry
title: "1625. Lexicographically Smallest String After Applying Operations"
permalink: "/leetcode/problem/2025-10-19-1625-lexicographically-smallest-string-after-applying-operations/"
leetcode_ui: true
entry_slug: "2025-10-19-1625-lexicographically-smallest-string-after-applying-operations"
---

[1625. Lexicographically Smallest String After Applying Operations](https://leetcode.com/problems/lexicographically-smallest-string-after-applying-operations/description) medium
[blog post](https://leetcode.com/problems/lexicographically-smallest-string-after-applying-operations/solutions/7285975/kotlin-rust-by-samoylenkodmitry-g5mp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05102025-1625-lexicographically-smallest?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/c4fZDlGP-Gs)

![29a9832f-f303-48a8-af19-682d15e0734e (1).webp](/assets/leetcode_daily_images/67ca6154.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1147

#### Problem TLDR

Min string after shifting by b and rotating odd indice by a #medium #bruteforce

#### Intuition

```j
    // s is small, up to 100
    // if b is even:    12345678  34567812  56781234  78123456
    //                  1234567   3456712   5671234   7123456 2345671 4567123  6712345  1234567
    // if b is odd:     12345678  45678123  78123456  23456781  56781234 81234567 3.. 6..
    //                  1234567   4567123   7123456   3456712  6712345  2345671 5671234
    // so only when b%2 == 0 && size%2 == 0 we don't have access to all elements
    // the size is small, we can brute-force every possible rotation to get the minimum
    // rotation of digit: 1, a = 3: 1,4,7,0,3,6,9,2
    // 25 minute: forgot about odd indices
    // 40 minute: we can't change indices independently ?
    // 43987654 b=3
    // *  *  *
    // 01234567
    // 34567012
    // 67012345
    // 12345670
    // 45670123
    // 70123456
    // 23456701
    // 56701234
    // 01234567  so, all indices can be on the first position
    // 50 minute: wrong steps calculation
    // 12345678901234 14, step=6
    // 58016941393090
    // 41393090580169

    //     * *     *
    // 123456789abcde
    // 56789abcde1234
```

Two situations:
1. odd length and odd shift: we can rotate only odd indices
2. otherwise we can rotate odd indices and even indices

#### Approach

* rotations should be for all indices; separate value of rotations for odd and even

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 50ms
    fun findLexSmallestString(s: String, a: Int, b: Int) = s.indices.minOf { i ->
        fun rot(c: Char) = (0..9).minBy { (c-'0'+it*a)%10 }
        var sh = s.drop((i*b)%s.length)+s.take((i*b)%s.length)
        val eo = listOf(if (s.length % 2 == 0 && b % 2 == 0) 0 else rot(sh[0]), rot(sh[1]))
        sh.mapIndexed { i,c -> '0'+(c-'0' + eo[i%2]*a)%10 }.joinToString("")
    }

```
```rust

// 0ms
    pub fn find_lex_smallest_string(s: String, a: i32, b: i32) -> String {
        (0..s.len()).map(|i| {
            let rot = |c: u8| {(0..10).min_by_key(|&x| (c - b'0' +  x * a as u8) % 10).unwrap()};
            let sh = s.chars().cycle().skip((i * b as usize)%s.len()).take(s.len()).collect::<String>();
            let eo = [if (s.len() as i32|b)&1 < 1 {0} else {rot(sh.as_bytes()[0])}, rot(sh.as_bytes()[1])];
            sh.bytes().enumerate().map(|(i, c)| (b'0' + (c - b'0' + eo[i&1]*a as u8)%10) as char).collect()
        }).min().unwrap()
    }

```

