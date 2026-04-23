---
layout: leetcode-entry
title: "907. Sum of Subarray Minimums"
permalink: "/leetcode/problem/2022-11-25-907-sum-of-subarray-minimums/"
leetcode_ui: true
entry_slug: "2022-11-25-907-sum-of-subarray-minimums"
---

[907. Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/) medium

```kotlin

    data class V(val v: Int, val count: Int)
    fun sumSubarrayMins(arr: IntArray): Int {
        val M = 1_000_000_007
        // 1 2 3 4 2 2 3 4
        //  1 2 3 2 2 2 3
        //   1 2 2 2 2 2
        //    1 2 2 2 2
        //     1 2 2 2
        //      1 2 2
        //       1 2
        //        1
        // f(1) = 1
        // f(2) = 2>1 ? f(1) + [1, 2]
        // f(3) = 3>2 ? f(2) + [1, 2, 3]
        // f(4) = 4>3 ? f(3) + [1, 2, 3, 4]
        // f(2) = 2<4 ? f(4) + [1, 2, 2, 2, 2] (1, 2, 3, 4 -> 3-2, 4-2, +2)
        // f(2) = 2=2 ? f(2) + [1, 2, 2, 2, 2, 2]
        // f(3) = 3>2 ? f(2) + [1, 2, 2, 2, 2, 2, 3]
        // f(4) = 4>3 ? f(3) + [1, 2, 2, 2, 2, 2, 3, 4]
        // 3 1 2 4    f(3) = 3    sum = 3  stack: [3]
        //  1 1 2     f(1): 3 > 1 , remove V(3,1), sum = sum - 3 + 1*2= 2, f=3+2=5, [(1,2)]
        //   1 1      f(2): 2>1, sum += 2 = 4, f+=4=9
        //    1       f(4): 4>2, sum+=4=8, f+=8=17
        val stack = Stack<V>()
        var f = 0
        var sum = 0
        arr.forEach { n ->
            var countRemoved = 0
            while (stack.isNotEmpty() && stack.peek().v > n) {
                val v = stack.pop()
                countRemoved += v.count
                var removedSum = (v.v*v.count) % M
                if (removedSum < 0) removedSum = M + removedSum
                sum = (sum - removedSum) % M
                if (sum < 0) sum = sum + M
            }
            val count = countRemoved + 1
            stack.add(V(n, count))
            sum = (sum + (n * count) % M) % M
            f = (f + sum) % M

        }
        return f
    }

```

First attempt is to build an N^2 tree of minimums, comparing adjacent elements row by row and finding a minimum.
That will take O(N^2) time and gives TLE.
Next observe that there is a repetition of the results if we computing result for each new element:
result = previous result + some new elements.
That new elements are also have a law of repetition:
sum = current element + if (current element < previous element) count of previous elements * current element else previous sum
We can use a stack to keep lowest previous elements, all values in stack must be less than current element.

O(N) time, O(N) space

