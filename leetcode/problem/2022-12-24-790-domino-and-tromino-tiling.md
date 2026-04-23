---
layout: leetcode-entry
title: "790. Domino and Tromino Tiling"
permalink: "/leetcode/problem/2022-12-24-790-domino-and-tromino-tiling/"
leetcode_ui: true
entry_slug: "2022-12-24-790-domino-and-tromino-tiling"
---

[790. Domino and Tromino Tiling](https://leetcode.com/problems/domino-and-tromino-tiling/description/) medium

[https://t.me/leetcode_daily_unstoppable/62](https://t.me/leetcode_daily_unstoppable/62)

[blog post](https://leetcode.com/problems/domino-and-tromino-tiling/solutions/2946811/kotlin-dfs-memo/)

```kotlin
  fun numTilings(n: Int): Int {
        val cache = Array<Array<Array<Long>>>(n) { Array(2) { Array(2) { -1L }}}
        fun dfs(pos: Int, topFree: Int, bottomFree: Int): Long {
            return when {
                pos > n -> 0L
                pos == n -> if (topFree==1 && bottomFree==1) 1L else 0L
                else -> {
                    var count = cache[pos][topFree][bottomFree]
                    if (count == -1L) {
                        count = 0L
                        when {
                            topFree==1 && bottomFree==1 -> {
                                count += dfs(pos+1, 1, 1) // vertical
                                count += dfs(pos+1, 0, 0) // horizontal
                                count += dfs(pos+1, 1, 0) // tromino top
                                count += dfs(pos+1, 0, 1) // tromino bottom
                            }
                            topFree==1 -> {
                                count += dfs(pos+1, 0, 0) // tromino
                                count += dfs(pos+1, 1, 0) // horizontal
                            }
                            bottomFree==1 -> {
                                count += dfs(pos+1, 0, 0) // tromino
                                count += dfs(pos+1, 0, 1) // horizontal
                            }
                        else -> {
                                count += dfs(pos+1, 1, 1) // skip
                            }
                        }

                        count = count % 1_000_000_007L
                    }
                    cache[pos][topFree][bottomFree] = count
                    count
                }
            }
        }
        return dfs(0, 1, 1).toInt()
    }

```

We can walk the board horizontally and monitor free cells. On each step, we can choose what figure to place. When end reached and there are no free cells, consider that a successful combination. Result depends only on the current position and on the top-bottom cell combination.* just do dfs+memo
* use array for a faster cache

Space: O(N), Time: O(N) - we only visit each column 3 times

