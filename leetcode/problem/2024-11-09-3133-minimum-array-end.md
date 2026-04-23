---
layout: leetcode-entry
title: "3133. Minimum Array End"
permalink: "/leetcode/problem/2024-11-09-3133-minimum-array-end/"
leetcode_ui: true
entry_slug: "2024-11-09-3133-minimum-array-end"
---

[3133. Minimum Array End](https://leetcode.com/problems/minimum-array-end/description/) medium
[blog post](https://leetcode.com/problems/minimum-array-end/solutions/6026342/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09112024-3133-minimum-array-end?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3HmHUjFnAaQ)
[deep-dive](https://notebooklm.google.com/notebook/dce0d628-9057-4387-94e2-3ca94ac72db3/audio)
![1.webp](/assets/leetcode_daily_images/52113858.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/795

#### Problem TLDR

`n`th of increasing sequence with `AND[..]=x` #medium #bit_manipulation

#### Intuition

Let's observe how we can form that sequence of increasing numbers:

```j

    //       x = 5
    // 0     101
    // 1     111
    // 2    1101 *
    // 3    1111
    // 4   10101
    // 5   10111
    // 6   11101
    // 7   11111
    // 8  100101 -> bit + x
    // 9  100111
    // 10 101101 *      n=10, first zero = 10 % 2 = 0, second zero = (10 / 2) % 2 = 1
    // 11 101111              third zero = (10 / 4) % 4
    // 12 110101
    //        ^ every other
    //      ^ every 2
    //     ^ every 4
    //    ^ every 8

```
Some observations:
* to `AND` operation resulting to `x`, all bits of `x` must be set in each number
* the minimum number is `x`
* we can only modify the vacant positions with `0` bits
* to form the next number we must alterate the vacant bit skipping the `1` bits
* in the `n`'th position each vacant bit is a `period % 2`, where period is a `1 << bit`
* another way to look at this: we have to add `(n-1)` inside the `0` bit positions of `x`

#### Approach

* one small optimization is to skip `1`-set bits with `trailing_ones()`

#### Complexity

- Time complexity:
$$O(log(n + x))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minEnd(n: Int, x: Int): Long {
        var period = n - 1; var a = x.toLong()
        for (b in 0..63) {
            if (period % 2 > 0) a = a or (1L shl b)
            if (b > 31 || (x shr b) % 2 < 1) period /= 2
        }
        return a
    }

```
```rust

    pub fn min_end(n: i32, x: i32) -> i64 {
        let (mut a, mut period, mut x, mut b) =
            (x as i64, (n - 1) as i64, x as i64, 0);
        while period > 0 {
            a |= (period & 1) << b;
            period >>= 1 - x & 1;
            let s = 1 + (x / 2).trailing_ones();
            x >>= s; b += s
        }
        a
    }

```
```c++

    long long minEnd(int n, int x) {
        long long a = x, y = x, period = (n - 1);
        for (int b = 0; period;) {
            a |= (period & 1LL) << b;
            period >>= 1 - y & 1;
            int s = 1 + __builtin_ctz(~(y / 2));
            y >>= s; b += s;
        }
        return a;
    }

```

