---
layout: leetcode-entry
title: "3130. Find All Possible Stable Binary Arrays II"
permalink: "/leetcode/problem/2026-03-10-3130-find-all-possible-stable-binary-arrays-ii/"
leetcode_ui: true
entry_slug: "2026-03-10-3130-find-all-possible-stable-binary-arrays-ii"
---

[3130. Find All Possible Stable Binary Arrays II](https://open.substack.com/pub/dmitriisamoilenko/p/10032026-3130-find-all-possible-stable?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) hard
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/10032026-3130-find-all-possible-stable?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10032026-3130-find-all-possible-stable?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Ccn-2zM23lc)

![86ab161b-a052-42a4-b55c-1517fb33ef65 (1).webp](/assets/leetcode_daily_images/8f3aa5eb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1293

#### Problem TLDR

01-array to make with z zeros, o ones, l repeats #hard #dp

#### Intuition

```j
    // so this is yesterday problem, but this time should be solved in O(n^2) instead of n^3
    // this is a n^3 solution: take every possible count 1..min(z,l)
    //
    // how to speed up?
    // dp[z][o]=dp[o][z-1]+dp[o][z-2]+...+dp[o][z-min(z,l)-1]+dp[o][z-min(z,l)]
    // dp[z-1][o]=         dp[o][z-2]+dp[o][z-3]+...+dp[o][z-1-min(z-1,l)-1]+dp[o][z-1 - min(z-1,l)]
    //
    // dp[z][o] = dp[z-1][o]+dp[o][z-1]+(?? dp[o][z-min(z,l)] ) // how to deal with tail?
    //
    //    z=3,l=2  dp[3][o]=dp[o][3-1]+dp[o][3-2]
    //             dp[3-1][o]=dp[o][3-1-1]+dp[o][3-1-2]
    //             dp[3][o]=dp[3-1][o] + dp[o][3-1] - dp[o][3-1-2]
    //
    //    z=3,l=4  dp[3][o]=dp[o][3-1]+dp[o][3-2]+dp[o][3-3]
    //             dp[3-1][o]=dp[o][3-1-1]+dp[o][3-1-2]
    //             dp[3][o]=dp[3-1][o] + dp[o][3-1]
    //
    //    z=5,l=3  dp[5][o]=dp[o][5-1]+dp[o][5-2]+dp[o][5-3]=dp[o][4]+dp[o][3]+dp[o][2]
    //             dp[4][o]=dp[o][4-1]+dp[o][4-2]+dp[o][4-3]=dp[o][3]+dp[o][2]+dp[o][1]
    //             dp[5][o]=dp[4][o]+dp[o][4]-dp[o][1]
    //

    // so if we rewrite to bottom up we can save one dimention of iteration
    //
```
* consider batches of repeating current digit
* alterate by swapping arguments (curr, other) = (other, curr-take)
* can take at most min(curr, limit)
* mathematically subtract dp[curr][other]-dp[curr-1][other] to see that for loop can be O(1)

#### Approach

* the combinatorics solution consists of: stars and bars (s-1 b-1), inclusion-exclusion principle f(curr)-f(excl), f(excl)-f(excl-excl)..., Fermat little theorem a^-1=a^(m-1) %m, modulo exponentiation a^y = a^2*(y/2)+a^2*(y%2)

#### Complexity

- Time complexity:
$$O(zo)$$

- Space complexity:
$$O(zo)$$

#### Code

```kotlin
// 366ms
    fun numberOfStableArrays(z: Int, o: Int, l: Int): Int {
        val dp = HashMap<Int, Long>(); val M = 1000000007
        fun dfs(z: Int, o: Int): Long = if (z<=0||o<0) 0L
        else if (o==0) {if (z <=l)1L else 0L} else dp.getOrPut(z*1000+o){
            (dfs(z-1,o)+dfs(o,z-1)-dfs(o,z-min(z,l)-1)+M)%M
        }
        return ((dfs(z, o) + dfs(o, z) + M) % M).toInt()
    }
```
```rust
// 39ms
    pub fn number_of_stable_arrays(z: i32, o: i32, l: i32) -> i32 {
        let (m,z,o,l) = (1000000007, z as usize, o as usize, l as usize);
        let (mut f, mut v) = ([1;2005], [1;2005]);
        for i in 1..2005 { f[i] = f[i-1] * i as i64 % m }
        let (mut b, mut e) = (f[2004], m-2);
        while e > 0 { if e & 1 == 1 { v[2004] = v[2004]*b%m} b=b*b%m; e/=2 }
        for i in (1..2005).rev() { v[i-1] = v[i]*i as i64 % m }
        let c = |n: usize, r: usize| if r > n {0} else {f[n]*v[r]%m*v[n-r]%m};
        let w = |n: usize, b: usize| if b == 0 {(n==0)as i64} else {
            (0..=b).take_while(|&i| n >= i*l + b).fold(0, |a,i|
                (a+[1,m-1][i&1]*c(b,i)%m*c(n-i*l-1,b-1)%m)%m )};
        (1..=z).fold(0, |a,k|
            (a + w(z,k)*(w(o,k)*2 + w(o,k-1) + w(o,k+1))%m)%m ) as _
    }
```

