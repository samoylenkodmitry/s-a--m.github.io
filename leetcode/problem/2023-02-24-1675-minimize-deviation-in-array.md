---
layout: leetcode-entry
title: "1675. Minimize Deviation in Array"
permalink: "/leetcode/problem/2023-02-24-1675-minimize-deviation-in-array/"
leetcode_ui: true
entry_slug: "2023-02-24-1675-minimize-deviation-in-array"
---

[1675. Minimize Deviation in Array](https://leetcode.com/problems/minimize-deviation-in-array/description/) hard

[blog post](https://leetcode.com/problems/minimize-deviation-in-array/solutions/3224614/kotlin-my-wrong-and-correct-intuition/)

```kotlin

fun minimumDeviation(nums: IntArray): Int {
    var minDiff = Int.MAX_VALUE
    with(TreeSet<Int>(nums.map { if (it % 2 == 0) it else it * 2 })) {
        do {
            val min = first()
            val max = pollLast()
            minDiff = minOf(minDiff, Math.abs(max - min))
            add(max / 2)
        } while (max % 2 == 0)
    }

    return minDiff
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/128
#### Intuition
We can notice, that the answer is the difference between the `min` and `max` from some resulting set of numbers.
My first (wrong) intuition was, that we can use two heaps for minimums and maximums, and only can divide by two from the maximum, and multiply by two from the minimum heap. That quickly transformed into too many edge cases.
The correct and tricky intuition: we can multiply all the numbers by 2, and then we can safely begin to divide all the maximums until they can be divided.

#### Approach
Use `TreeSet` to quickly access to the `min` and `max` elements.

#### Complexity
- Time complexity:
$$O(n(log_2(n) + log_2(h)))$$, where h - is a number's range
- Space complexity:
$$O(n)$$

