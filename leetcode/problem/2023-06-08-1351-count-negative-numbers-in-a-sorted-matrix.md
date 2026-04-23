---
layout: leetcode-entry
title: "1351. Count Negative Numbers in a Sorted Matrix"
permalink: "/leetcode/problem/2023-06-08-1351-count-negative-numbers-in-a-sorted-matrix/"
leetcode_ui: true
entry_slug: "2023-06-08-1351-count-negative-numbers-in-a-sorted-matrix"
---

[1351. Count Negative Numbers in a Sorted Matrix](https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/description/) easy
[blog post](https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/solutions/3611472/kotlin-fold/)
[substack](https://dmitriisamoilenko.substack.com/p/08062023-1351-count-negative-numbers?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/239
#### Problem TLDR
Count negatives in a sorted by row and by column matrix.
#### Intuition
Consider example:

```

4  3  2 -1
3  2  1 -1
1  1 -1 -2
^ we are here
-1 -1 -2 -3

```

If we set position `x` at the first negative number, it is guaranteed, that the next `row[x]` will also be negative. So we can skip already passed columns.
#### Approach
Let's use Kotlin's `fold` operator.
#### Complexity
- Time complexity:
$$O(n + m)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun countNegatives(grid: Array<IntArray>): Int =
    grid.fold(0 to 0) { (total, prev), row ->
        var curr = prev
        while (curr < row.size && row[row.lastIndex - curr] < 0) curr++
        (total + curr) to curr
    }.first
}

```

