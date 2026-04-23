---
layout: leetcode-entry
title: "3573. Best Time to Buy and Sell Stock V"
permalink: "/leetcode/problem/2025-12-17-3573-best-time-to-buy-and-sell-stock-v/"
leetcode_ui: true
entry_slug: "2025-12-17-3573-best-time-to-buy-and-sell-stock-v"
---

[3573. Best Time to Buy and Sell Stock V](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-v/description/) medium
[blog post](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-v/solutions/7419480/kotlin-rust-by-samoylenkodmitry-uw7q/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17122025-3573-best-time-to-buy-and?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rfHEQbAUo5A)

![81ebf892-5407-48cf-b3ef-044f06651049 (1).webp](/assets/leetcode_daily_images/9e73411c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1207

#### Problem TLDR

K best deals buy/short/sell stocks #medium #dp

#### Intuition

* the O(n^3) is accepted in Kotlin, dp state is (i,k) and inner loop to close the deal
* optimized version: use (i,k,s) as a state, s=0 -free, s=1 - bought, s=1 - short the stoks

#### Approach

* write DFS, then rewrite to iterative version
* then we can space optimize if necessary

#### Complexity

- Time complexity:
$$O(n^2)$$, or O(n^3) with inner search

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
// 3791ms
    fun maximumProfit(p: IntArray, k: Int): Long {
        val dp = Array(p.size) { LongArray(p.size)}
        fun dfs(i: Int, k: Int): Long = if (i==p.size || k==0)0L else
        dp[i][k].takeIf { it > 0}?:{
            val skip = dfs(i+1, k)
            val start = (i+1..<p.size).maxOfOrNull { j -> abs(p[i]-p[j]) + dfs(j+1, k-1) }?:0L
            dp[i][k] = max(start, skip); dp[i][k]
        }()
        return dfs(0,k)
    }
```
```rust
// 61ms
    pub fn maximum_profit(p: Vec<i32>, k: i32) -> i64 {
        let n = p.len(); let k = k as usize;
        let mut dp = vec![vec![vec![0i64; n+1]; k+1]; 3];
        for i in (0..=n).rev() { for kk in 0..=k { for s in -1..=1 {
            let idx = (s + 1) as usize;
            dp[idx][kk][i] = if i == n || kk == 0 { 0 } else {
                let skip = dp[idx][kk][i+1];
                if s == 0 {
                    let buy = dp[2][kk][i+1] - p[i] as i64;
                    let sell = dp[0][kk][i+1] + p[i] as i64;
                    if i == n - 1 { skip } else { buy.max(sell).max(skip) }
                } else {
                    let close = dp[1][kk-1][i+1] + (s * p[i] as i64);
                    if i == n - 1 { close } else { close.max(skip) }
                }
        }}}} dp[1][k][0]
    }
```

