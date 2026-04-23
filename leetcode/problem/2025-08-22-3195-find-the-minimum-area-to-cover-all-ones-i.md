---
layout: leetcode-entry
title: "3195. Find the Minimum Area to Cover All Ones I"
permalink: "/leetcode/problem/2025-08-22-3195-find-the-minimum-area-to-cover-all-ones-i/"
leetcode_ui: true
entry_slug: "2025-08-22-3195-find-the-minimum-area-to-cover-all-ones-i"
---

[3195. Find the Minimum Area to Cover All Ones I](https://leetcode.com/problems/find-the-minimum-area-to-cover-all-ones-i/description/) medium
[blog post](https://leetcode.com/problems/find-the-minimum-area-to-cover-all-ones-i/solutions/7109329/kotlin-rust-by-samoylenkodmitry-sws5/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22082025-3195-find-the-minimum-area?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/CbUfPIs2Z18)

![1.webp](/assets/leetcode_daily_images/3b8e9c14.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1089

#### Problem TLDR

Min all-1 rectangle #medium

#### Intuition

Compute 4 variables: minX..maxX, minY..maxY

#### Approach

* or, we can go from the corners

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 945ms
    fun minimumArea(g: Array<IntArray>): Int {
        var a = 0; var b = 0; var c = g.size; var d = g[0].size
        for (y in 0..<c) for (x in g[0].indices) if (g[y][x] > 0)
            { a = y; b = max(b, x); c = min(c, y); d = min(d, x) }
        return (b-d+1)*(a-c+1)
    }

```
```kotlin

// 1019ms
    fun minimumArea(g: Array<IntArray>): Int {
        var w = g[0].size; var h = g.size
        for (y in g.indices) if (1 !in g[y]) --h else break
        for (y in g.lastIndex downTo 0) if (1 !in g[y]) --h else break
        for (x in g[0].indices) if (g.indices.all { g[it][x] < 1}) --w else break
        for (x in g[0].lastIndex downTo 0) if (g.indices.all { g[it][x] < 1}) --w else break
        return w * h
    }

```
```rust

// 39ms
    pub fn minimum_area(g: Vec<Vec<i32>>) -> i32 {
        let mut r = [0, 0, g.len(), g[0].len()];
        for y in 0..r[2] { for x in 0..g[0].len() { if g[y][x] > 0
            { r = [y, r[1].max(x), r[2].min(y), r[3].min(x)] }
        }} ((r[1] - r[3] + 1) * (r[0] - r[2] + 1)) as _
    }

```
```c++

// 275ms
    int minimumArea(vector<vector<int>>& g) {
        int a=0,b=0,c=size(g),d=size(g[0]),n=c,m=d;
        for (int y = 0; y < n; ++y) for (int x = 0; x < m; ++x)
            g[y][x]&&(a=y,b=b>x?b:x,c=c<y?c:y,d=d<x?d:x);
        return (a-c+1)*(b-d+1);
    }

```
```python

// 3144ms
    def minimumArea(_, g):
       i, j = zip(*((y,x) for y,r in enumerate(g) for x,v in enumerate(r) if v))
       return (max(j)-min(j)+1)*(max(i)-min(i)+1)

```
```python

// 2593ms
    def minimumArea(_, g):
        n,m = len(g), len(g[0])
        t = next(i for i in range(n) if 1 in g[i])
        b = next(i for i in range(n-1,-1,-1) if 1 in g[i])
        l = next(j for j in range(m) if any(g[i][j] for i in range(n)))
        r = next(j for j in range(m-1,-1,-1) if any(g[i][j] for i in range(n)))
        return (b-t+1)*(r-l+1)

```

