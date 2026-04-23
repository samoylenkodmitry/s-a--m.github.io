---
layout: leetcode-entry
title: "Basic Calculator"
permalink: "/leetcode/problem/2022-11-20-basic-calculator/"
leetcode_ui: true
entry_slug: "2022-11-20-basic-calculator"
---

[https://leetcode.com/problems/basic-calculator/](https://leetcode.com/problems/basic-calculator/) hard

```

    fun calculate(s: String): Int {
        var i = 0
        var sign = 1
        var eval = 0
        while (i <= s.lastIndex) {
            val chr = s[i]
            if (chr == '(') {
                //find the end
                var countOpen = 0
                for (j in i..s.lastIndex) {
                    if (s[j] == '(') countOpen++
                    if (s[j] == ')') countOpen--
                    if (countOpen == 0) {
                        //evaluate substring
                        eval += sign * calculate(s.substring(i+1, j)) // [a b)
                        sign = 1
                        i = j
                        break
                    }
                }
            } else if (chr == '+') {
                sign = 1
            } else if (chr == '-') {
                sign = -1
            } else if (chr == ' ') {
                //nothing
            } else {
                var num = (s[i] - '0').toInt()
                for (j in (i+1)..s.lastIndex) {
                    if (s[j].isDigit()) {
                        num = num * 10 + (s[j] - '0').toInt()
                        i = j
                    } else  break
                }
                eval += sign * num
                sign = 1
            }
            i++
        }
        return eval
    }

```

This is a classic calculator problem, nothing special.
* be careful with the indexes

Complexity: O(N)
Memory: O(N), because of the recursion, worst case is all the input is brackets

