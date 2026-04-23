---
layout: leetcode-entry
title: "1751. Maximum Number of Events That Can Be Attended II"
permalink: "/leetcode/problem/2023-07-15-1751-maximum-number-of-events-that-can-be-attended-ii/"
leetcode_ui: true
entry_slug: "2023-07-15-1751-maximum-number-of-events-that-can-be-attended-ii"
---

[1751. Maximum Number of Events That Can Be Attended II](https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended-ii/description/) hard
[blog post](https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended-ii/solutions/3766779/kotln-dp/)
[substack](https://dmitriisamoilenko.substack.com/p/15072023-1751-maximum-number-of-events?sd=pf)
![image.png](/assets/leetcode_daily_images/56170b8d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/276

#### Problem TLDR

Max sum of at most `k` `values` from non-intersecting array of `(from, to, value)` items

#### Intuition

Let's observe example:

```bash
        // 0123456789011
        // [     4 ]
        // [1][2][3][2]
        //      [4][2]

```

If `k=1` we choose `[4]`
if `k=2` we choose `[4][2]`
if `k=3` we choose `[2][3][2]`

###### What will not work:

* sweep line algorithm, as it is greedy, but there is an only `k` items we must choose and we must do backtracking
* adding to Priority Queue and popping the lowest values: same problem, we must backtrack

###### What will work:

* asking for a hint: this is what I used
* full search: at every `index` we can `pick` or `skip` the element
* sorting: it will help to reduce irrelevant combinations by doing a Binary Search for the next non-intersecting element

We can observe, that at any given position the result only depends on the suffix array. That means we can safely cache the result by the current position.

#### Approach

For more robust Binary Search code:
* use inclusive `lo`, `hi`
* check the last condition `lo == hi`
* always write the result `next = mid`
* always move the borders `lo = mid + 1`, `hi = mid - 1`

#### Complexity

- Time complexity:
$$O(nklog(n))$$

- Space complexity:
$$O(nk)$$

#### Code

```kotlin

    fun maxValue(events: Array<IntArray>, k: Int): Int {
        // 0123456789011
        // [     4 ]
        // [1][2][3][2]
        //      [4][2]
        val inds = events.indices.sortedWith(compareBy({ events[it][0] }))
        // my ideas:
        // sort - good
        // sweep line ? - wrong
        // priority queue ? - wrong
        // binary search ? 1..k - wrong
        // used hints:
        // hint: curr + next vs drop  dp?
        // hint: binary search next
        val cache = mutableMapOf<Pair<Int, Int>, Int>()
        fun dfs(curr: Int, canTake: Int): Int {
          return if (curr ==  inds.size || canTake == 0) 0
          else cache.getOrPut(curr to canTake) {
            val (_, to, value) = events[inds[curr]]
            var next = inds.size
            var lo = curr + 1
            var hi = inds.lastIndex
            while (lo <= hi) {
              val mid = lo + (hi - lo) / 2
              val (nextFrom, _, _) = events[inds[mid]]
              if (nextFrom > to) {
                next = mid
                hi = mid - 1
              } else lo = mid + 1
            }
            maxOf(value + dfs(next, canTake - 1), dfs(curr + 1, canTake))
          }
        }
        return dfs(0, k)
    }

```

