---
layout: leetcode-entry
title: "Remove All Adjacent Duplicates In String"
permalink: "/leetcode/problem/2022-11-10-remove-all-adjacent-duplicates-in-string/"
leetcode_ui: true
entry_slug: "2022-11-10-remove-all-adjacent-duplicates-in-string"
---

[https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/) easy

Solution:

```

    fun removeDuplicates(s: String): String {
        val stack = Stack<Char>()
        s.forEach { c ->
            if (stack.isNotEmpty() && stack.peek() == c) {
                stack.pop()
            } else {
                stack.push(c)
            }
        }
        return stack.joinToString("")
    }

```

Explanation: Just scan symbols one by one and remove duplicates from the end.
Complexity: O(N)
Memory: O(N)

