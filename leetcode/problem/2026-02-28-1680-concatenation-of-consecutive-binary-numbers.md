---
layout: leetcode-entry
title: "1680. Concatenation of Consecutive Binary Numbers"
permalink: "/leetcode/problem/2026-02-28-1680-concatenation-of-consecutive-binary-numbers/"
leetcode_ui: true
entry_slug: "2026-02-28-1680-concatenation-of-consecutive-binary-numbers"
---

[1680. Concatenation of Consecutive Binary Numbers](https://dmitriisamoilenko.substack.com/publish/posts/detail/189443191/share-center) medium
[blog post](https://dmitriisamoilenko.substack.com/publish/posts/detail/189443191/share-center)
[substack](https://dmitriisamoilenko.substack.com/publish/posts/detail/189443191/share-center)
[youtube](https://youtu.be/A8od0NdRN2o)

![img](/assets/leetcode_daily_images/866f190f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1283

#### Problem TLDR

Concat binaries 1..n #medium

#### Intuition

Brute-force is accepted.

The log solution:
r = r*2^L + x
x = x + 1
repeated for distinct groups of lengths
group divider is (1 shl L) - 1, count = curr-prev
convert to matrix
(2^L 1 0)
(  0 1 1)
(  0 0 1)

RX1 = M^count * RX1

exponentiation: a^b = a^(b/2)*2 + a^(b)%2

#### Approach

* for brute-force: we can reuse previous length and increase on each 2^x
* we can use countLeadingZeroBits()

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 127ms
    fun concatenatedBinary(n: Int) = (1..n).fold(0L)
    { r, x -> (x + (r shl (32-x.countLeadingZeroBits())))%1000000007 }
```
```rust
// 2ms
    pub fn concatenated_binary(n: i32) -> i32 {
        const I: [i64; 9] = [1, 0, 0, 0, 1, 0, 0, 0, 1]; const M:i64 = 1000000007;
        let (n, mut r, mut s, mut l) = (n as i64, 0, 1, 1);
        fn mul(a: [i64; 9], b: [i64; 9]) -> [i64; 9] {
            from_fn(|i| (0..3).map(|k| a[i/3*3+k] * b[k*3+i%3]).sum::<i64>() % M)
        }
        fn pow(b: [i64; 9], p: i64) -> [i64; 9] {
            if p == 0 { I } else { mul(pow(mul(b, b), p / 2), if p % 2 > 0 { b } else { I }) }
        }
        while s <= n {
            let e = n.min((1i64 << l) - 1);
            let t = pow([(1i64 << l) % M, 1, 0, 0, 1, 1, 0, 0, 1], e - s + 1);
            r = (r * t[0] + s * t[1] + t[2]) % M;
            s = e + 1; l += 1;
        }
        r as _
    }
```

