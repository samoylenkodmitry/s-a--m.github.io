---
layout: leetcode-entry
title: "1267. Count Servers that Communicate"
permalink: "/leetcode/problem/2025-01-23-1267-count-servers-that-communicate/"
leetcode_ui: true
entry_slug: "2025-01-23-1267-count-servers-that-communicate"
---

[1267. Count Servers that Communicate](https://leetcode.com/problems/count-servers-that-communicate/description/) medium
[blog post](https://leetcode.com/problems/count-servers-that-communicate/solutions/6319064/kotlin-rust-by-samoylenkodmitry-ttza/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23012025-1267-count-servers-that?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UH2zMpxL424)
![1.webp](/assets/leetcode_daily_images/dc1cb6cf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/873

#### Problem TLDR

Connected servers by row or column #medium

#### Intuition

The brute force is accepted.

Some optimizations: we can count rows and columns frequency, then scan servers with any freq > 1.

Another way is to us Union-Find.

#### Approach

* let's golf in Kotlin

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(n + m)$$

#### Code

```kotlin

    fun countServers(g: Array<IntArray>) = g
        .flatMap { r -> r.indices.map { r to it }}
        .count { (r, x) -> r[x] * g.sumOf { it[x] } * r.sum() > 1 }

```
```rust

    pub fn count_servers(grid: Vec<Vec<i32>>) -> i32 {
        let (mut rs, mut cs, mut vs) =
          (vec![0; grid.len()], vec![0; grid[0].len()], vec![]);
        for y in 0..grid.len() { for x in 0..grid[0].len() {
            if grid[y][x] > 0 { rs[y] += 1; cs[x] += 1; vs.push((y, x)) }
        }}
        vs.into_iter().filter(|&(y, x)| rs[y] > 1 || cs[x] > 1).count() as i32
    }

```
```c++

    int countServers(vector<vector<int>>& g) {
        int r[250], c[250]; int s = 0;
        for (int y = 0; y < size(g); ++y)
            for (int x = 0; x < size(g[0]); ++x)
                g[y][x] && (++r[y], ++c[x]);
        for (int y = 0; y < size(g); ++y)
            for (int x = 0; x < size(g[0]); ++x)
                s += g[y][x] * r[y] * c[x] > 1;
        return s;
    }

```

