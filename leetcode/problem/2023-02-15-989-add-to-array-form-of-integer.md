---
layout: leetcode-entry
title: "989. Add to Array-Form of Integer"
permalink: "/leetcode/problem/2023-02-15-989-add-to-array-form-of-integer/"
leetcode_ui: true
entry_slug: "2023-02-15-989-add-to-array-form-of-integer"
---

[989. Add to Array-Form of Integer](https://leetcode.com/problems/add-to-array-form-of-integer/description/) easy

[blog post](https://leetcode.com/problems/add-to-array-form-of-integer/solutions/3188017/kotlin-single-pass/)

```kotlin
    fun addToArrayForm(num: IntArray, k: Int): List<Int> {
        var carry = 0
        var i = num.lastIndex
        var n = k
        val res = LinkedList<Int>()
        while (i >= 0 || n > 0 || carry > 0) {
            val d1 = if (i >= 0) num[i--] else 0
            val d2 = if (n > 0) n % 10 else 0
            var d = d1 + d2 + carry
            res.addFirst(d % 10)
            carry = d / 10
            n = n / 10
        }
        return res
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/119
#### Intuition
Iterate from the end of the array and calculate sum of `num % 10`, `carry` and `num[i]`.

#### Approach
* use linked list to add to the front of the list in O(1)
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$

