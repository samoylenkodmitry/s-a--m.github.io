---
layout: leetcode-entry
title: "2897. Apply Operations on Array to Maximize Sum of Squares"
permalink: "/leetcode/problem/2026-10-12-2897-apply-operations-on-array-to-maximize-sum-of-squares/"
leetcode_ui: true
entry_slug: "2026-10-12-2897-apply-operations-on-array-to-maximize-sum-of-squares"
---

[2897. Apply Operations on Array to Maximize Sum of Squares](https://leetcode.com/problems/apply-operations-on-array-to-maximize-sum-of-squares/solutions/8568646/kotlin-rust-by-samoylenkodmitry-dco2/) hard
[substack](https://dmitriisamoilenko.substack.com/p/12102026-2897-apply-operations-on?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/N_KQo3nXA2Y)

https://dmitrysamoylenko.com/leetcode/

![12.10.2026.webp](/assets/leetcode_daily_images/12.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1510

#### Problem TLDR

Max sum of squares after any pairwise a=a|b, b=a&b

#### Intuition

Didn't solved myself.

The bitwise property is that total number of ones is zeros is preserved:
101 | 110 = 111
101 & 110 = 100
total 4 ones 4 zeros
This is also moves all ones to the one number and all zeros to the other number.
That means bits can be sorted across all the numbers an we can construct full-bit numbers at the one end of the array, then peek k from them.

#### Approach

* should have meditated on bits a little bit longer

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun maxSum(n: List<Int>, k: Int)=IntArray(32){b->n.count{it and (1 shl b)>0}}
    .let { f -> (1..k).fold(0L) { r, _ ->
        val x = (0..31).fold(0L) {x,b->if(f[b]-->0) x or (1L shl b) else x}
        (r + x*x) % 1000000007
    }}
```
```rust
    pub fn max_sum(n: Vec<i32>, k: i32) -> i32 {
        let mut f = [0; 32]; for x in n { for b in 0..32 {f[b]+=(x>>b&1) as usize}}
        (0..k).fold(0i64, |r, _| {
            let x = (0..32).fold(0i64,|x,b|if f[b] > 0 {f[b]-=1; x|1<<b} else {x});
            (r + x * x) % 1_000_000_007
        }) as i32
    }
```

