---
layout: leetcode-entry
title: "3742. Maximum Path Score in a Grid"
permalink: "/leetcode/problem/2026-04-30-3742-maximum-path-score-in-a-grid/"
leetcode_ui: true
entry_slug: "2026-04-30-3742-maximum-path-score-in-a-grid"
---

[3742. Maximum Path Score in a Grid](https://leetcode.com/problems/maximum-path-score-in-a-grid/solutions/8115942/kotlin-rust-by-samoylenkodmitry-7mxe/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30042026-3742-maximum-path-score?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RkNnxydfrek)

https://dmitrysamoylenko.com/leetcode/

![30.04.2026.webp](/assets/leetcode_daily_images/30.04.2026.webp)
#### Join me on Telegram

#### Problem TLDR

Max right-bottom path value with at most k cost

#### Intuition

Recursive DP: at each cell pick max between right and bottom.
Bottom-up DP: at each cell check all costs up and left.

#### Approach

* return some big negative value
* bottom-up: continue the costs c in 0..k curr[c+cost] = max(L,T) + value

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n^3 | n^2)$$

#### Code

```kotlin
    fun maxPathScore(g: Array<IntArray>, k: Int): Int {
        val dp = HashMap<Int, Int>()
        fun dfs(x:Int, y:Int, c:Int): Int = dp.getOrPut(x*40000+y*200+c) {
            val nc = c+((g.getOrNull(y)?.getOrNull(x)?:4000)+1)/2;
            if (nc > k) -99999 else g[y][x] +
            if (x==g[0].size-1&&y==g.size-1)0 else max(dfs(x+1,y,nc),dfs(x,y+1,nc))
        }
        return dfs(0, 0, 0).takeIf{it>=0} ?:-1
    }
```
```rust
    pub fn max_path_score(g: Vec<Vec<i32>>, k: i32) -> i32 {
        let (k, mut dp) = (k as usize, vec![vec![-1;k as usize+1];g[0].len()]);
        for (y,r) in g.iter().enumerate() { for (x,&v) in r.iter().enumerate() {
            let (cost, mut curr) = (((v+1)/2)as usize, vec![-1; k+1]);
            for c in 0..(k+1).saturating_sub(cost) {
                let p = if x+y<1&&c<1{0} else { dp[x][c].max(if x>0{dp[x-1][c]}else{-1})};
                if p >= 0 { curr[c+cost] = v + p }
            }
            dp[x] = curr
        }} *dp[dp.len()-1].iter().max().unwrap()
    }
```

