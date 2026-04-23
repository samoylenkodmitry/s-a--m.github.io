---
layout: leetcode-entry
title: "1992. Find All Groups of Farmland"
permalink: "/leetcode/problem/2024-04-20-1992-find-all-groups-of-farmland/"
leetcode_ui: true
entry_slug: "2024-04-20-1992-find-all-groups-of-farmland"
---

[1992. Find All Groups of Farmland](https://leetcode.com/problems/find-all-groups-of-farmland/description/) medium
[blog post](https://leetcode.com/problems/find-all-groups-of-farmland/solutions/5048640/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20042024-1992-find-all-groups-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cYm87NiqL2A)
![2024-04-20_09-05.webp](/assets/leetcode_daily_images/b78c2807.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/577

#### Problem TLDR

Count `1`-rectangles in `0-1` 2D matrix #medium

#### Intuition

We can use DFS or just move bottom-right, as by task definition all `1`-islands are rectangles

#### Approach

* find the right border, then fill arrays with zeros
* Rust didn't have a `fill` method

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(r)$$, where `r` is a resulting count of islands, can be up to `nm/2`

#### Code

```kotlin

    fun findFarmland(land: Array<IntArray>) = buildList {
        for (y in land.indices) for (x in land[0].indices) { if (land[y][x] > 0) {
            var y2 = y; var x2 = x
            while (x2 < land[0].size && land[y][x2] > 0) x2++
            while (y2 < land.size && land[y2][x] > 0) land[y2++].fill(0, x, x2)
            add(intArrayOf(y, x, y2 - 1, x2 - 1))
    }}}.toTypedArray()

```
```rust

    pub fn find_farmland(mut land: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut res = vec![];
        for y in 0..land.len() { for x in 0..land[0].len() { if land[y][x] > 0 {
            let (mut y2, mut x2) = (y, x);
            while x2 < land[0].len() && land[y][x2] > 0 { x2 += 1 }
            while y2 < land.len() && land[y2][x] > 0 {
                for i in x..x2 { land[y2][i] = 0 }
                y2 += 1
            }
            res.push(vec![y as i32, x as i32, y2 as i32 - 1, x2 as i32 - 1])
        }}}; res
    }

```

