---
layout: leetcode-entry
title: "Remove Duplicates From Sorted Array"
permalink: "/leetcode/problem/2022-11-11-remove-duplicates-from-sorted-array/"
leetcode_ui: true
entry_slug: "2022-11-11-remove-duplicates-from-sorted-array"
---

[https://leetcode.com/problems/remove-duplicates-from-sorted-array/](https://leetcode.com/problems/remove-duplicates-from-sorted-array/) easy

Just do what is asked. Keep track of the pointer to the end of the "good" part.

```

    fun removeDuplicates(nums: IntArray): Int {
        var k = 0
        for (i in 1..nums.lastIndex) {
            if (nums[k] != nums[i]) nums[++k] = nums[i]
        }

        return k + 1
    }

```

Complexity: O(N)
Memory: O(1)

