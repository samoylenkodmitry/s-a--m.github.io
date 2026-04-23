---
layout: leetcode-entry
title: "1915. Number of Wonderful Substrings"
permalink: "/leetcode/problem/2024-04-30-1915-number-of-wonderful-substrings/"
leetcode_ui: true
entry_slug: "2024-04-30-1915-number-of-wonderful-substrings"
---

[1915. Number of Wonderful Substrings](https://leetcode.com/problems/number-of-wonderful-substrings/description/) medium
[blog post](https://leetcode.com/problems/number-of-wonderful-substrings/solutions/5090753/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30042024-1915-number-of-wonderful?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/o0RD8uGFhQA)
![2024-04-30_09-16.webp](/assets/leetcode_daily_images/1f7e89d3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/588

#### Problem TLDR

Count substrings with at most one odd frequency #medium #bit_manipulation

#### Intuition

This is a hard problem.
Let's try to look at the problem with our bare hands:

```j
    // aba
    // a     a
    // ab    -
    //  b    b
    //  ba   -
    // aba   aba

    // aab
    // a     +    xor = a
    // aa    +    xor = 0
    //  a    +    xor = a
    // aab   +    xor = b
    //  ab   -    xor = ab
    //   b   +    xor = b
    //   * = (aa, a) + b

    // dp or two-pointers?
    // dp: f(aabb) = f(aab)? + b
    // two pointers:
    // aabb
    //    i  move i: a + a + b + b + aa + aab + aabb
    //    j  move j: abb + bb
    //  skip ab?
```
We quickly run out of possible solutions patterns: neither dp or two pointers approach would work.
However, there are some thoughts:
* only odd-even matters, so, we can somehow use `xor`
* `xor` works well for interval `i..j` when we pre-compute all the prefixes: `xor i..j = xor 0..j xor xor 0..i`

This is where my brain has stopped, and I used the hints:

* use prefix's bitmask, as we only have `10` unique chars

Let's try to make use of the prefix's bitmasks:

```j

    // bitmask           00
    // a                 01
    //  a                00
    //   b               10  m[ab] = m[aab] xor m[a]
    //    b              00  m[abb] = m[aabb] xor m[a]
    //     how many previous masks have mismatched bits?
    //                                  ~~~~~~~~~~
```
We know the current prefix's bitmask `m` and our interest is how many subarrays on the left are good. We can xor with all the previous masks to find out the xor result of subarrays: this result must have at most one `1` bit. We can compress this search by putting unique masks in a counter HashMap.

```j
    // mismatched = differs 1 bit or equal
    //
    // ab                m
    //                   00
    // a                 01 +1(00)
    //  b                11 +1(01)

    // 0123
    // aabb              m   res
    //                   00
    //0a                 01  +1(00)
    //1 a                00  +2(00,01)
    //2  b               10  +2(00,00)
    //3   b              00  +4(00,01,00,10)
    //

```

#### Approach

* Another neat trick: we don't have to check all the masks from a HashMap, just check by changing every of the `10` bits of mask.
* array is faster, we have at most `2^10` unique bits combinations

### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(2^k)$$, k - is an alphabet, at most 2^10 masks total

#### Code

```kotlin

    fun wonderfulSubstrings(word: String): Long {
        val masksCounter = LongArray(1024); masksCounter[0] = 1
        var m = 0; var res = 0L
        for (c in word) {
            m = m xor (1 shl (c.code - 'a'.code))
            res += masksCounter[m]
            for (i in 0..9) res += masksCounter[m xor (1 shl i)]
            masksCounter[m]++
        }
        return res
    }

```
```rust

    pub fn wonderful_substrings(word: String) -> i64 {
        let mut counter = vec![0; 1024]; counter[0] = 1;
        let (mut m, mut res) = (0, 0);
        for b in word.bytes() {
            m ^= 1 << (b - b'a');
            res += counter[m];
            for i in 0..10 { res += counter[m ^ (1 << i)] }
            counter[m] += 1
        }; res
    }

```

