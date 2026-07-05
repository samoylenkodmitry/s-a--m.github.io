---
layout: leetcode-entry
title: "1301. Number of Paths with Max Score"
permalink: "/leetcode/problem/2026-07-05-1301-number-of-paths-with-max-score/"
leetcode_ui: true
entry_slug: "2026-07-05-1301-number-of-paths-with-max-score"
---

[1301. Number of Paths with Max Score](https://leetcode.com/problems/number-of-paths-with-max-score/solutions/8377126/kotlin-rust-by-samoylenkodmitry-stq5/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05072026-1301-number-of-paths-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Z_SaG0xrlEk)

https://dmitrysamoylenko.com/leetcode/

![05.07.2026.webp](/assets/leetcode_daily_images/05.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1411

#### Problem TLDR

Count maximal paths

#### Intuition

Use Dp = answer for the current cell

#### Approach

* use size+1 to avoid 'if' checks
* we can use just previous & current row for dp to make space O(n)

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
    fun pathsWithMaxScore(b: List<String>): IntArray {
        val n = b.size-1; val d = List(n+2){Array(n+2){intArrayOf(0,0)}}; d[n][n][1] = 1
        for (y in n downTo 0) for (x in n downTo 0) if (b[y][x] !in "XS") {
            val l = listOf(d[y][x+1], d[y+1][x], d[y+1][x+1]); val m = l.maxOf { it[0] }
            val c = l.filter { it[0] == m }.fold(0) { r, i -> (r + i[1]) % 1000000007 }
            if (c > 0) d[y][x] = intArrayOf(m + if (b[y][x]=='E') 0 else b[y][x] - '0', c)
        }
        return d[0][0]
    }
```
```rust
    pub fn paths_with_max_score(b: Vec<String>) -> Vec<i32> {
        let n = b.len(); let mut p = [[0; 2]; 102]; p[n][1] = 1;
        for y in (0..n).rev() { let mut c = [[0; 2]; 102]; for x in (0..n).rev() {
            let v = b[y].as_bytes()[x]; if v == 88 { continue }
            let m = c[x + 1][0].max(p[x][0]).max(p[x + 1][0]);
            let k = [c[x+1], p[x], p[x+1]].iter().filter(|i|i[0]==m).fold(0,|r,i|(r+i[1])%1000000007);
            if k > 0 { c[x] = [m + if v > 60 { 0 } else { (v - 48) as i32 }, k]; }
        } p = c } p[0].into()
    }
```

