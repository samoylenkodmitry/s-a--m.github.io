---
layout: leetcode-entry
title: "2300. Successful Pairs of Spells and Potions"
permalink: "/leetcode/problem/2023-04-02-2300-successful-pairs-of-spells-and-potions/"
leetcode_ui: true
entry_slug: "2023-04-02-2300-successful-pairs-of-spells-and-potions"
---

[2300. Successful Pairs of Spells and Potions](https://leetcode.com/problems/successful-pairs-of-spells-and-potions/description/) medium

[blog post](https://leetcode.com/problems/successful-pairs-of-spells-and-potions/solutions/3369146/kotlin-binary-search/)

```kotlin

fun successfulPairs(spells: IntArray, potions: IntArray, success: Long): IntArray {
    potions.sort()
    return IntArray(spells.size) { ind ->
        var lo = 0
        var hi = potions.lastIndex
        var minInd = potions.size
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2
            if (potions[mid].toLong() * spells[ind].toLong() >= success) {
                minInd = minOf(minInd, mid)
                hi = mid - 1
            } else lo = mid + 1
        }
        potions.size - minInd
    }
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/168
#### Intuition
If we sort `potions`, we can find the lowest possible value of `spell[i]*potion[i]` that is `>= success`. All other values are bigger by the math multiplication property.
#### Approach
* sort `potions`
* binary search the `lowest` index
* use `long` to solve the integer overflow
###### For more robust binary search code:
* use inclusive `lo` and `hi`
* do the last check `lo == hi`
* always compute the result `minInd`
* always move the `lo` and the `hi`
* safely compute `mid` to not overflow
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(n)$$

