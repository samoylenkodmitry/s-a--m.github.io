---
layout: leetcode-entry
title: "3756. Concatenate Non-Zero Digits and Multiply by Sum II"
permalink: "/leetcode/problem/2026-07-08-3756-concatenate-non-zero-digits-and-multiply-by-sum-ii/"
leetcode_ui: true
entry_slug: "2026-07-08-3756-concatenate-non-zero-digits-and-multiply-by-sum-ii"
---

[3756. Concatenate Non-Zero Digits and Multiply by Sum II](https://leetcode.com/problems/concatenate-non-zero-digits-and-multiply-by-sum-ii/solutions/8383748/kotlin-rust-by-samoylenkodmitry-ahxx/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08072026-3756-concatenate-non-zero?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7Hjaw3eIMik)

https://dmitrysamoylenko.com/leetcode/

![08.07.2026.webp](/assets/leetcode_daily_images/08.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1414

#### Problem TLDR

Queries of sum digits * concat digits

#### Intuition

```j
    // prefix power?
    // 1   a
    // 123 b
    // l r
    //  23
    // 1   (*100?) (could be * 10^10^5)
    // b - a * 10^(r-l), should it be modPow?
```
Separate prefix sums and prefix powers. Track prefix lengths.

#### Approach

* use longs to avoid overflow on multiplications

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun sumAndMultiply(S: String, q: Array<IntArray>) = (S.length+1).let {
        val M = 1_000_000_007L; val x = LongArray(it); val w = IntArray(it)
        val s = LongArray(it); val p = LongArray(it){1}
        S.forEachIndexed { i, c ->
            x[i+1] = if (c > '0') (x[i] * 10 + (c-'0')) % M else x[i]
            s[i+1] = s[i] + (c-'0'); w[i+1] = w[i] + (c-'0').sign; p[i+1] = p[i]*10%M
        }
        q.map {(l,r)->(x[r+1] - x[l] * p[w[r+1]-w[l]] % M + M) % M * (s[r+1]-s[l]) % M}
    }
```
```rust
    pub fn sum_and_multiply(s: String, q: Vec<Vec<i32>>) -> Vec<i32> {
        let m = 1_000_000_007i64; let mut v = vec![(0, 0, 0, 1)];
        for d in s.bytes().map(|b| (b - 48) as i64) {
            let &(px, pw, pa, pp) = v.last().unwrap();
            v.push((if d > 0 {(px*10+d)%m} else {px}, pw+(d>0) as usize, pa+d, pp*10%m));
        }
        q.iter().map(|q| {
            let (l, r) = (v[q[0] as usize], v[q[1] as usize + 1]); let pw = v[r.1-l.1].3;
            ((r.0 + m - l.0 * pw % m) % m * (r.2 - l.2) % m) as i32
        }).collect()
    }
```

