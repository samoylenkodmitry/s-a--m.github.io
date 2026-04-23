---
layout: leetcode-entry
title: "3459. Length of Longest V-Shaped Diagonal Segment"
permalink: "/leetcode/problem/2025-08-27-3459-length-of-longest-v-shaped-diagonal-segment/"
leetcode_ui: true
entry_slug: "2025-08-27-3459-length-of-longest-v-shaped-diagonal-segment"
---

[3459. Length of Longest V-Shaped Diagonal Segment](https://leetcode.com/problems/length-of-longest-v-shaped-diagonal-segment/description) hard
[blog post](https://leetcode.com/problems/length-of-longest-v-shaped-diagonal-segment/solutions/7127269/kotlin-by-samoylenkodmitry-z63p/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27082025-3459-length-of-longest-v?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/65p4eQxap-w)

![1.webp](/assets/leetcode_daily_images/8e58ca1d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1094

#### Problem TLDR

Max length of 1-2-0-2.. diagonal sequence, on cw rotation #hard #dp

#### Intuition

Do Depth-First Search, add memoization (however accepted without it)

#### Approach

* enumerate diagonals as 0..3, cw rotation is `(d+1)%4`, ccw `(d+3)%4`
* v is irrelevant to dp key
* dp key can be Int: `(500*y+x)*100 + dir*10 + rot`
* I think the HashMap is the weakest point of performance

#### Complexity

- Time complexity:
$$O(mn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 1219ms
    fun lenOfVDiagonal(g: Array<IntArray>): Int {
        val nxy = arrayOf(-1,-1,1,1,-1); val nv = arrayOf(2,2,0); var res = 0
        fun dfs(y: Int, x: Int, dir: Int, rot: Int, v: Int): Int =
            if (y !in 0..<g.size || x !in 0..<g[0].size || g[y][x] != v ) 0
            else 1 + max(dfs(y+nxy[dir],x+nxy[dir+1],dir,rot,nv[v]),
                         if (rot > 0) dfs(y,x,(dir+1)%4,0,v)-1 else 0)
        for (y in g.indices) for (x in g[0].indices) if (g[y][x] == 1)
            res = max(res, (0..3).maxOf {dfs(y,x,it,1,1)})
        return res
    }

```
```kotlin

// 981ms
    fun lenOfVDiagonal(g: Array<IntArray>): Int {
        val nxy = arrayOf(-1,-1,1,1,-1); val nv = arrayOf(2,2,0); var res = 0
        val dp = HashMap<Int, Int>()
        fun dfs(y: Int, x: Int, dir: Int, rot: Int, v: Int): Int =
            if (y < 0 || x < 0 || y == g.size || x  == g[0].size || g[y][x] != v ) 0
            else 1 + dp.getOrPut((y*500 + x)*100 + dir*10 + rot) { max(
                dfs(y+nxy[dir],x+nxy[dir+1],dir,rot,nv[v]),
                if (rot > 0) dfs(y,x,(dir+1)%4,0,v)-1 else 0) }
        for (y in g.indices) for (x in g[0].indices) if (g[y][x] == 1)
            res = max(res, (0..3).maxOf {dfs(y,x,it,1,1)})
        return res
    }

```

