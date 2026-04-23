---
layout: leetcode-entry
title: "71. Simplify Path"
permalink: "/leetcode/problem/2023-04-12-71-simplify-path/"
leetcode_ui: true
entry_slug: "2023-04-12-71-simplify-path"
---

[71. Simplify Path](https://leetcode.com/problems/simplify-path/description/) medium

```kotlin

fun simplifyPath(path: String): String =
"/" + Stack<String>().apply {
    path.split("/").forEach {
        when (it) {
            ".." -> if (isNotEmpty()) pop()
            "." -> Unit
            "" -> Unit
            else -> push(it)
        }
    }
}.joinToString("/")

```

[blog post](https://leetcode.com/problems/simplify-path/solutions/3407165/kotlin-stack/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-12042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/178
#### Intuition
We can simulate what each of the `.` and `..` commands do by using a `Stack`.
#### Approach
* split the string by `/`
* add elements to the Stack if they are not commands and not empty
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

