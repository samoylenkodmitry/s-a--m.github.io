---
layout: leetcode-entry
title: "1411. Number of Ways to Paint N \u00d7 3 Grid"
permalink: "/leetcode/problem/2026-12-03-1411-number-of-ways-to-paint-n-3-grid/"
leetcode_ui: true
entry_slug: "2026-12-03-1411-number-of-ways-to-paint-n-3-grid"
---

[1411. Number of Ways to Paint N × 3 Grid](https://leetcode.com/problems/number-of-ways-to-paint-n-3-grid/description) hard
[blog post](https://leetcode.com/problems/number-of-ways-to-paint-n-3-grid/solutions/7461704/kotlin-rust-by-samoylenkodmitry-2d1q/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03122026-1411-number-of-ways-to-paint?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Zn-xUbW4Z_E)

![9d53a2e8-6017-47e7-aa21-83a85dfd80e1 (1).webp](/assets/leetcode_daily_images/150a0549.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1224

#### Problem TLDR

Ways to color 3xN grid #hard #dp

#### Intuition

Do DFS + dp. The state is current cell and 3 previous colors.

#### Approach

* optimization 1: there are only two patterns A -> 2A+2B, B -> 2A+3B
* optimization 2: exponentiation matrix for O(log(n)) solution. Skipped this, too hard to implement

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 1ms
    fun numOfWays(n: Int): Long {
        var a = 6L; var b = a; val M = 1000000007
        for (i in 2..n) { a += (a+b+b)%M; b += a }
        return (a+b)%M
    }
```
```rust
// 20ms
    pub fn num_of_ways(n: i32) -> i32 {
        let mut dp = vec![-1;(n*192) as usize];
        fn dfs(dp: &mut [i32], i: i32, m: i32, n: i32) -> i32 {
            if i == n { return 1 }; let k = ((i<<6)|m) as usize; if dp[k] != -1 { return dp[k] }
            let res = ((0..3).map(|c|
                if i%3>0 && c==m&3 || i>=3&&c==m>>4 {0} else { dfs(dp,i+1,((m<<2)|c)&63,n)as i64}
            ).sum::<i64>()%1000000007) as i32; dp[k] = res; res
        }
        dfs(&mut dp, 0, 0, 3*n)
    }
```

