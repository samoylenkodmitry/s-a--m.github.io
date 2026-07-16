---
layout: leetcode-entry
title: "3867. Sum of GCD of Formed Pairs"
permalink: "/leetcode/problem/2026-07-16-3867-sum-of-gcd-of-formed-pairs/"
leetcode_ui: true
entry_slug: "2026-07-16-3867-sum-of-gcd-of-formed-pairs"
---

[3867. Sum of GCD of Formed Pairs](https://leetcode.com/problems/sum-of-gcd-of-formed-pairs/solutions/8400378/kotlin-rust-by-samoylenkodmitry-1v69/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16072026-3867-sum-of-gcd-of-formed?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6WixMY2VDxs)

https://dmitrysamoylenko.com/leetcode/

![16.07.2026.webp](/assets/leetcode_daily_images/16.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1422

#### Problem TLDR

Sum of gcd(min,max) of gcd(x, prefix max)

#### Intuition

The entire algorithm is give, just implement.

#### Approach

* remember gcd as `a/b, bab`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun gcdSum(n: IntArray) = run {
        var max = 0; fun gcd(a: Int, b: Int): Int = if (b==0)a else gcd(b,a%b)
        val g = n.map { max=max(max,it);gcd(max, it) }.sorted()
        (0..<g.size/2).sumOf { i -> 1L*gcd(g[i],g[g.size-1-i]) }
    }
```
```rust
    pub fn gcd_sum(n: Vec<i32>) -> i64 {
        let g = |mut a, mut b| { while b != 0 { (a, b) = (b, a % b); } a };
        let mut m = 0;
        let s: Vec<_> = n.iter().map(|&x| { m = m.max(x); g(m, x) }).sorted().collect();
        (0..s.len() / 2).map(|i| g(s[i], s[s.len() - 1 - i]) as i64).sum()
    }
```

