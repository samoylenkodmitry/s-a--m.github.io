---
layout: leetcode-entry
title: "3655. XOR After Range Multiplication Queries II"
permalink: "/leetcode/problem/2026-04-09-3655-xor-after-range-multiplication-queries-ii/"
leetcode_ui: true
entry_slug: "2026-04-09-3655-xor-after-range-multiplication-queries-ii"
---

[3655. XOR After Range Multiplication Queries II]() hard
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/09042026-3655-xor-after-range-multiplication?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/QNmDemnPMwc)

![09.04.2026.webp](/assets/leetcode_daily_images/09.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1323

#### Problem TLDR

Run stepped multiplication queries #hard #mod

#### Intuition

Almost solved, gave up.
```j
    // same as yesterday
    // i learned it before: split queries by sqrt(k)
    // a^x = a^{2*(x/2 + x%2)} = (a*a)^{x/2}*(a*a)^{x%2}
    // 18 minute, there is an error somewhere
    // 28 minute, 2 errors fixes, still wrong answer
    // nums =
    //[2,3,1,5,4]
    //queries =
    //[[1,4,2,3],[0,2,1,2]]
    // 4, 18, 2, 15, 4 -- expected
    //
    // 2,  3, 1,  5, 4
    // 4,  6, 2,  5, 4   (after k=1 l..r=0..2 v=2)
    // 4, 18, 6, 15, 12 -- mine (after k=2 l..r=1..4 v=3) (looks like i multiplied all without step)
    //     *      *
    // 42 minute: how to deal with step?
    // 45 minute: another test case wrong 68/605
    // 49 minute: i don't see any error in my code
    //
    // nums =
    // [562,62]
    // queries =
    // [[0,1,2,7],[1,1,2,11],[0,1,2,2],[1,1,1,11],[1,1,2,1],[0,0,1,9],[0,1,2,4],[1,1,1,6],[0,0,2,17]]
    // Output
    // 591836426
    // Expected
    // 4839076
    // 58 minute gave up.
```

1. Divide problem on slow/fast path: k = sqrt(q) is the point of separation
2. fast path is the big k steps, can be brute forced
3. slow path: save events of start-stop for every k in p[k][l..r] individually
4. collect them for every k separately
5. modulo doesnt allow 1/v operation, have to use pow(v, M-2) instead

#### Approach

* my point of failure was the attention to one detail: when to cancel, it is not r+1, we have to use steps
* the threshold of 40: ~300ms vs 400: ~1300ms

#### Complexity

- Time complexity:
$$O(nsqrt(q))$$

- Space complexity:
$$O(nsqrt(q))$$

#### Code

```kotlin
// 323ms
    fun xorAfterQueries(n: IntArray, q: Array<IntArray>): Int {
        val p = Array(40) { IntArray(n.size+1){1} }; val M = 1_000_000_007L
        fun pow(a: Long, b: Long): Long = if (b==0L) 1L else pow(a*a%M, b/2)*(if (b%2>0)a else 1L)%M
        for ((l,r,k,v) in q)
            if (k >= 40) for (i in l..r step k) n[i] = (1L*n[i]*v%M).toInt() else {
                p[k][l] = (1L*p[k][l]*v%M).toInt(); val next = l + ((r-l)/k+1)*k
                if (next < n.size) p[k][next] = (1L*p[k][next]*pow(1L*v, M-2L)%M).toInt()
            }
        for (k in 1..<40) for (i in n.indices) {
            if (i >= k) p[k][i] = (1L*p[k][i]*p[k][i-k]%M).toInt()
            n[i] = (1L*n[i]*p[k][i]%M).toInt()
        }
        return n.reduce(Int::xor)
    }
```
```rust
// 322ms
    pub fn xor_after_queries(mut n: Vec<i32>, q: Vec<Vec<i32>>) -> i32 {
        const M: i64 = 1000000007; let z = n.len(); let mut c = vec![vec![1; z + 1]; 40];
        fn p(a: i64, b: i64) -> i64 { if b < 1 { 1 } else { p(a * a % M, b / 2) * (if b % 2 > 0 { a } else { 1 }) % M } }
        for u in q {
            let (l, r, k, v) = (u[0] as usize, u[1] as usize, u[2] as usize, u[3] as i64);
            if k < 40 { c[k][l] = c[k][l] * v % M; let x = r - (r - l) % k + k; if x < z { c[k][x] = c[k][x] * p(v, M - 2) % M } }
            else { for i in (l..=r).step_by(k) { n[i] = (n[i] as i64 * v % M) as i32 } }
        }
        for k in 1..40 { for i in 0..z {
            if i >= k { c[k][i] = c[k][i] * c[k][i - k] % M }
            n[i] = (n[i] as i64 * c[k][i] % M) as i32;
        }}
        n.into_iter().fold(0, |a, b| a ^ b)
    }
```

