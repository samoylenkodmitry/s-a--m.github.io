---
layout: leetcode-entry
title: "2444. Count Subarrays With Fixed Bounds"
permalink: "/leetcode/problem/2023-03-04-2444-count-subarrays-with-fixed-bounds/"
leetcode_ui: true
entry_slug: "2023-03-04-2444-count-subarrays-with-fixed-bounds"
---

[2444. Count Subarrays With Fixed Bounds](https://leetcode.com/problems/count-subarrays-with-fixed-bounds/description/) hard

[blog post](https://leetcode.com/problems/count-subarrays-with-fixed-bounds/solutions/3255030/kotlin-nlogn-but-not-tricky-solution-optimal/)

```kotlin

fun countSubarrays(nums: IntArray, minK: Int, maxK: Int): Long {
    val range = minK..maxK
    var i = 0
    var sum = 0L
    if (minK == maxK) {
        var count = 0
        for (i in 0..nums.lastIndex) {
            if (nums[i] == minK) count++
            else count = 0
            if (count > 0) sum += count
        }
        return sum
    }
    while (i < nums.size) {
        val curr = nums[i]
        if (curr in range) {
            val minInds = TreeSet<Int>()
                val maxInds = TreeSet<Int>()
                    var end = i
                    while (end < nums.size && nums[end] in range) {
                        if (nums[end] == minK) minInds.add(end)
                        else if (nums[end] == maxK) maxInds.add(end)
                        end++
                    }
                    if (minInds.size > 0 && maxInds.size > 0) {
                        var prevInd = i - 1
                        while (minInds.isNotEmpty() && maxInds.isNotEmpty()) {
                            val minInd = minInds.pollFirst()!!
                            val maxInd = maxInds.pollFirst()!!
                            val from = minOf(minInd, maxInd)
                            val to = maxOf(minInd, maxInd)
                            val remainLenAfter = (end - 1 - to).toLong()
                            val remainLenBefore = (from - (prevInd + 1)).toLong()
                            sum += 1L + remainLenAfter + remainLenBefore + remainLenAfter * remainLenBefore
                            prevInd = from
                            if (to == maxInd) maxInds.add(to)
                            else if (to == minInd) minInds.add(to)
                        }
                    }
                    if (i == end) end++
                    i = end
                } else i++
            }
            return sum
        }
and more clever solution:
fun countSubarrays(nums: IntArray, minK: Int, maxK: Int): Long {
    var sum = 0L
    if (minK == maxK) {
        var count = 0
        for (i in 0..nums.lastIndex) {
            if (nums[i] == minK) count++
            else count = 0
            if (count > 0) sum += count
        }
        return sum
    }
    val range = minK..maxK
    // 0 1 2 3 4 5 6 7 8 91011
    // 3 7 2 2 2 2 2 1 2 3 2 1
    //   b
    //               *...*
    //                   *...*
    var border = -1
    var posMin = -1
    var posMax = -1
    for (i in 0..nums.lastIndex) {
        when (nums[i]) {
            !in range -> border = i
            minK -> posMin = i
            maxK -> posMax = i
        }
        if (posMin > border && posMax > border)
        sum += minOf(posMin, posMax) - border
    }
    return sum
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/137
#### Intuition
First thought is that we can observe only subarrays, where all the elements are in a range `min..max`. Next, there are two possible scenarios:
1. If `minK==maxK`, our problem is a trivial count of the combinations, $$ 0 + 1 + .. + (n-1) + n = n*(n+1)/2$$
2. If `minK != maxK`, we need to take every `minK|maxK` pair, and count how many items are in range `before` them and how many `after`. Then, as we observe the pattern of combinations:

```

// 0 1 2 3 4 5 6    min=1, max=3
// ------------------
// 1 2 3 2 1 2 3
// 1 2 3          *** 0..2 remainLenAfter = 6 - 2 = 4
// 1 2 3 2
// 1 2 3 2 1
// 1 2 3 2 1 2
// 1 2 3 2 1 2 3
//     3 2 1      *** 2..4 remainLenAfter = 6 - 4 = 2
//     3 2 1 2
//     3 2 1 2 3
//   2 3 2 1               remainLenBefore = 2 - (0 + 1) = 1, sum += 1 + remainLenAfter += 1+2 += 3
//   2 3 2 1 2
//   2 3 2 1 2 3
//         1 2 3  *** 4..6 remainLenBefore = 4 - 4 + 1 = 1
//       2 1 2 3

// 1 2 1 2 3 2 3
// *.......*      *** 0..4 sum += 1 + 2 = 3
//     *...*      *** 2..4 rla = 6 - 4 = 2, rlb = 2 - (0 + 1) = 1, sum += 1 + rla + rlb + rlb*rla += 6 = 9

// 1 3 5 2 7 5
// *...*
//

```

we derive the formula: $$sum += 1 + suffix + prefix + suffix*prefix$$

A more clever, but less understandable solution: is to count how many times we take a condition where we have a `min` and a `max` and each time add `prefix` count. Basically, it is the same formula, but with a more clever way of computing. (It is like computing a combination sum by adding each time the counter to sum).
#### Approach

For the explicit solution, we take each interval, store positions of the `min` and `max` in a `TreeSet`, then we must take poll those mins and maxes and consider each range separately:

```

// 3 2 3 2 1 2 1
// *.......*
//     *...*

// 3 2 1 2 3 2 1
// *...*
//     *...*
//         *...*

// 3 2 1 2 1 2 3
// *...*
//     *.......*
//         *...*

// 3 2 1 2 3 3 3
// *...*
//     *...*

// 3 2 2 2 2 2 1
// *...........*

// 1 1 1 1 1 1 1
// *.*
//   *.*
//     *.*
//       *.*
//         *.*
//           *.*

```

For the tricky one solution, just see what other clever man already wrote on the leetcode site and hope you will not get the same problem in an interview.

#### Complexity

- Time complexity:
$$O(nlog_2(n))$$ -> $$O(n)$$

- Space complexity:
$$O(n)$$ -> $$O(1)$$

