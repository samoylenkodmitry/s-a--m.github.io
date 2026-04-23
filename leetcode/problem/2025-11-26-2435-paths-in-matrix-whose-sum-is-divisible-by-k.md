---
layout: leetcode-entry
title: "2435. Paths in Matrix Whose Sum Is Divisible by K"
permalink: "/leetcode/problem/2025-11-26-2435-paths-in-matrix-whose-sum-is-divisible-by-k/"
leetcode_ui: true
entry_slug: "2025-11-26-2435-paths-in-matrix-whose-sum-is-divisible-by-k"
---

[2435. Paths in Matrix Whose Sum Is Divisible by K](https://leetcode.com/problems/paths-in-matrix-whose-sum-is-divisible-by-k/description/) hard
[blog post](https://leetcode.com/problems/paths-in-matrix-whose-sum-is-divisible-by-k/solutions/7375461/kotlin-rust-by-samoylenkodmitry-62qr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26112025-2435-paths-in-matrix-whose?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xUM0vYox4e4)

![92e260af-fc15-4c08-97d6-25914946ca75 (1).webp](/assets/leetcode_daily_images/1101c2d1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1185

#### Problem TLDR

Paths in 2D matrix %k #hard

#### Intuition

Each cell hash exactly `k` possible remainders of all paths, so memo by [y][x][k]

#### Approach

* top down is just a memo+bruteforce DFS

#### Complexity

- Time complexity:
$$O(nk)$$

- Space complexity:
$$O(nk)$$

#### Code

```kotlin
// 479ms
    fun numberOfPaths(g: Array<IntArray>, k: Int): Int {
        val M = 1000000007; val dp = HashMap<Int, Int>()
        fun dfs(y: Int, x: Int, s: Int): Int =
            if (y==g.size-1&&x==g[0].size-1) {if ((g[y][x]+s)%k==0) 1 else 0}
            else if (y==g.size||x==g[0].size) 0 else
            dp.getOrPut(y*1000000+x*100+s) {
                (dfs(y+1, x, (g[y][x] + s)%k) + dfs(y, x+1, (g[y][x]+s)%k))%M
            }
        return dfs(0, 0, 0)
    }
```
```rust
// 29ms
    pub fn number_of_paths(g: Vec<Vec<i32>>, k: i32) -> i32 {
        let k = k as usize; let mut r = vec![vec![0; k+1]; g[0].len()+1];
        let mut n = r.clone(); r[1][0] = 1;
        for row in g {
            for x in 0..r.len()-1 { let v = row[x]as usize; for i in 0..k {
                n[x+1][(i+v)%k] = (n[x][i] + r[x+1][i])%1000000007 }}
            (r,n)=(n,r)
        }; r[r.len()-1][0]
    }
```

