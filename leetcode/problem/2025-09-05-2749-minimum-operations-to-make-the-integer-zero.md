---
layout: leetcode-entry
title: "2749. Minimum Operations to Make the Integer Zero"
permalink: "/leetcode/problem/2025-09-05-2749-minimum-operations-to-make-the-integer-zero/"
leetcode_ui: true
entry_slug: "2025-09-05-2749-minimum-operations-to-make-the-integer-zero"
---

[2749. Minimum Operations to Make the Integer Zero](https://leetcode.com/problems/minimum-operations-to-make-the-integer-zero/description/) medium
[blog post](https://leetcode.com/problems/minimum-operations-to-make-the-integer-zero/solutions/7157930/kotlin-rust-by-samoylenkodmitry-khqr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05092025-2749-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/uUFaN-Ak-00)

![1.webp](/assets/leetcode_daily_images/826ffdc4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1103

#### Problem TLDR

Min ops to subtract n2+2^(0..60) from n1 and make n1=0 #medium

#### Intuition

Didn't solve.

```j
    // n1 = a*n2 + b*2^i, i=0..60
    // count of (2^i) == a
    //           n1 = c*2^0+d*2^1+..x*2^32
    //                bits
    //  2^0 = 1
    //  2^1 = 2
    // 2^i always positive
    //
    // n1=4  n2=0   2^2
    // n1=3  n2=0   2^1,2^0
    // n1=3  n2=1   2^1+1
    // n1=3  n2=2   2^0+2
    // n1=3  n2=3   -1
    // n1=7  n2=1   2^2+1,2^0+1
    // n1=7  n2=-1  2^3-1
    //    7-n2=8 2^3
    // n1=8  n2=-1
    //    8-n2=9 9-2^3=1
    //    1-n2=2 2-2^1=0
    // try just subtract the highest bit
    // n1=3 n2=-2
    //    3-n2=5 5-2^2=1
    //    1-n2=3 3-2^1=1   didn't work that way
    //
    // 1 2 4 8 16 32
    // look for hints (23 minute)
    // hint one if n2==0, we need just countOneBits operations
    // hint two if n1 can be 0, we need at most 60 ops (why?)
    //
    // n1 = a*n2 + b*2^i, i=0..60
    // n1-a*n2 = sum_a(2^i)  (i up to a)
    //          0b01000

```

* do arithmetics: `n1-a*n2 = sum(2^x) = Y`
* a_max is `x=0 for all 2^x = Y`
* a_min is `countOneBits` - optimal exponentiation

#### Approach

* the hardest step is the grasping of `min..max` for `a` and understanding that we *can* split number if the `a` in that range

#### Complexity

- Time complexity:
$$O(60log(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 4ms
    fun makeTheIntegerZero(n1: Int, n2: Int) = (1..32)
    .firstOrNull { it in (n1-1L*it*n2).countOneBits()..n1-1L*it*n2 } ?: -1

```
```rust

// 0ms
    pub fn make_the_integer_zero(n1: i32, n2: i32) -> i32 {
        let (mut x, mut r) = (n1 as i64, 0);
        for k in 1..33 {
            x -= n2 as i64;
            if x < k { return -1 }
            if k >= x.count_ones() as i64 { return k as _ }
        } -1
    }

```
```c++

// 0ms
    int makeTheIntegerZero(int n1, int n2) {
        for(long long k = 1, x = n1;;++k) {
            x -= n2;
            if (x < k) return -1;
            if (__builtin_popcountll(x) <= k) return k;
        }
    }

```
```python

// 3ms
    def makeTheIntegerZero(_, n1, n2):
        return next((t for t in range(1, 33)
            if (n1 - t*n2).bit_count() <= t <= n1 - t*n2), -1)

```

