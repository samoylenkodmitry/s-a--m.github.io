---
layout: leetcode-entry
title: "3534. Path Existence Queries in a Graph II"
permalink: "/leetcode/problem/2026-07-10-3534-path-existence-queries-in-a-graph-ii/"
leetcode_ui: true
entry_slug: "2026-07-10-3534-path-existence-queries-in-a-graph-ii"
---

[3534. Path Existence Queries in a Graph II](https://leetcode.com/problems/path-existence-queries-in-a-graph-ii/solutions/8388003/kotlin-by-samoylenkodmitry-7zxo/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10072026-3534-path-existence-queries?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tJYqTupC-3s)

https://dmitrysamoylenko.com/leetcode/

![10.07.2026.webp](/assets/leetcode_daily_images/10.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1416

#### Problem TLDR

Queries of connected nodes shortest distances

#### Intuition

Didn't solved.
```j
    // 1 1  1 1 1  1 1  1 1 1 1 from every node to every node dist = 1
    // 1 2 . 4 5 6 7 8 9 md=2
    // * * .
    // 1 1 .
    //   * . *
    // 2 1 . 0
    //     . * *
    // 2 2 . 1 0
    //     . * * *
    // 2 2 . 1 1 0
    // 3 2 . 2 1 1 0
    // 3 3 . 2 2 1 1 0
    // 3 3 . 2 2 2 1 1 0
    // 4 3 . 3 2 2 2 1 1 0
    //     ^
    //     can be removed, distances stay the same
    //     so the distance is (a-b)/md

    // 1 hr mark: wrong answer: 91, 92, 127, 173, 179, 182 md=51, 91-182 my 2, correct 3
    //                          so this simple formula doesnt work
    //                          182-91=91; 91/51 = 2
    //                          but 91+51 = 142, so we go to 127
    //                          127+51 = 178, we have to go to 173
    //                          173+51=200+ we arrive at 182 at 3 steps
    //                          that means for each number we should track next reachable
    //                          or do dp[current_position][steps_required]=reachable_position
    //                          but this is O(n^2), so let's give up
    // hints: binary jumping (?)

```
* sort the numbers
* use sliding window to find the rightmost jump for every cell: left goest +1, right goes until diff is bigger than max
* prepare binary lifting jump table: u[k][x] = u[k-1][Y] where Y = u[k-1][x]; for each x we prepare all 2^k (0..31) jumps by reusing previous;
* in query: find the left 'c' pointer and the right 't' pointer positions
* by moving the left 'c' pointer with jump table count the jumps st += 2^k if it is not overshoot the right 't' pointer u[k][c]<t

#### Approach

* learn the binary lifting

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun pathExistenceQueries(n: Int, ns: IntArray, md: Int, qs: Array<IntArray>)=run{
        val s = ns.sorted(); val u = Array(18){IntArray(n)}; var r = 0
        for (l in 0..<n) { while (r+1<n && s[r+1]-s[l]<=md)++r; u[0][l]=r}
        for (k in 1..17) for (x in 0..<n) u[k][x] = u[k-1][u[k-1][x]]
        qs.map {(a,b) -> if (a==b)return@map 0
            var c = s.binarySearch(min(ns[a],ns[b])); val t = s.binarySearch(max(ns[a],ns[b]))
            if (u[17][c]<t) return@map -1; var st = 1
            for (k in 17 downTo 0) if (u[k][c]<t) {c = u[k][c]; st += 1 shl k }; st
        }
    }
```
```rust

```

